import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv

from langchain_ollama.llms import OllamaLLM

from google import genai
from openai import OpenAI

load_dotenv()

OLLAMA_MODEL = "translategemma:4b"
GEMINI_MODEL = "gemini-2.5-flash"
OPENAI_MODEL = "gpt-5-nano"

# Tek istekte çevrilecek en fazla balon; üstü okuma sırası korunarak parçalanır.
# Küçük modelde (4b) çok uzun prompt kaliteyi düşürür, bu yüzden sınır var.
BATCH_SIZE = 40
# Parçalar arası bağlam için bir önceki parçanın son N çevirisi prompta eklenir
CONTEXT_TAIL = 3

TRANSLATE_SYSTEM = (
    "You are a professional English (en) to Turkish (tr) comic/manga translator. "
    "You translate whole chapters at once so the dialogue stays consistent: "
    "same character names, same terminology and a consistent tone across all bubbles. "
    "If series notes / a glossary is provided, follow it strictly."
)

BATCH_INSTRUCTIONS = (
    "Below are the speech bubbles of one chapter, in reading order. "
    "Translate each numbered line into natural, fluent Turkish.\n"
    "Rules:\n"
    "- Output ONLY the numbered translations, one per line, using the same numbers.\n"
    "- Never merge, skip or reorder lines.\n"
    "- No explanations, no notes, no extra text."
)

_LINE_RE = re.compile(r"^\s*(\d+)\s*[.):\-]\s*(.*)$")


def _parse_numbered(raw: str, expected: int) -> Dict[int, str]:
    """'1. çeviri' biçimindeki çıktıyı {numara: metin} sözlüğüne çevirir."""
    out: Dict[int, str] = {}
    cur: Optional[int] = None
    for line in (raw or "").splitlines():
        s = line.strip()
        if not s or s.startswith("```"):
            continue
        m = _LINE_RE.match(s)
        if m:
            idx = int(m.group(1))
            if 1 <= idx <= expected:
                cur = idx
                out[idx] = m.group(2).strip()
                continue
        # Numarasız devam satırı: bir önceki çeviriye ekle
        if cur is not None:
            out[cur] = (out[cur] + " " + s).strip()
    return out


def _clean_single(raw: str) -> str:
    s = (raw or "").strip()
    s = s.replace("```", "").strip()
    s = " ".join(s.split())
    s = s.strip('"').strip()
    # Modelin başa koyabileceği '1.' artığını sil
    m = _LINE_RE.match(s)
    if m:
        s = m.group(2).strip()
    return s.strip('"').strip()


class LLM:
    # ── Sağlayıcı çağrıları ───────────────────────────────────
    def _call_ollama(self, prompt: str) -> str:
        model = OllamaLLM(model=OLLAMA_MODEL)
        return str(model.invoke(prompt)).strip()

    def _call_gemini(self, prompt: str) -> str:
        api_key = os.getenv("gemini_api_key")
        if not api_key:
            raise RuntimeError("gemini_api_key bulunamadı (.env).")
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        return (response.text or "").strip()

    def _call_openai(self, prompt: str, system: str = TRANSLATE_SYSTEM) -> str:
        api_key = os.getenv("openai_api_key")
        if not api_key:
            raise RuntimeError("openai_api_key bulunamadı (.env).")
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        return (response.choices[0].message.content or "").strip()

    def _complete(self, prompt: str, system: str = TRANSLATE_SYSTEM) -> str:
        """Ollama -> Gemini -> OpenAI sırasıyla dener, ilk dolu cevabı döndürür."""
        for name, fn in (
            ("ollama", lambda: self._call_ollama(system + "\n\n" + prompt)),
            ("gemini", lambda: self._call_gemini(system + "\n\n" + prompt)),
            ("openai", lambda: self._call_openai(prompt, system)),
        ):
            try:
                out = fn()
                if out:
                    return out
            except Exception as e:
                print(f"  LLM hatası ({name}): {e}")
        return ""

    # ── Prompt kurulumu ───────────────────────────────────────
    @staticmethod
    def _glossary_block(series_md: str) -> str:
        series_md = (series_md or "").strip()
        if not series_md:
            return ""
        return (
            "Series notes and glossary (FOLLOW STRICTLY for names, terms and tone):\n"
            "-----\n" + series_md + "\n-----\n\n"
        )

    # ── Çeviri ────────────────────────────────────────────────
    def translate_single(self, text: str, series_md: str = "") -> str:
        text = (text or "").strip()
        if not text:
            return ""
        prompt = (
            self._glossary_block(series_md)
            + "Translate the following speech bubble into natural Turkish. "
              "Output ONLY the translation.\n\n" + text
        )
        return _clean_single(self._complete(prompt))

    def translate_batch(self, texts: Sequence[str], series_md: str = "") -> List[str]:
        """Bölümün tüm balonlarını tek seferde (gerekirse parçalı) çevirir.

        Dönen liste girişle aynı uzunluktadır; çevrilemeyen satır kaynak
        metniyle doldurulur.
        """
        texts = [(t or "").strip() for t in texts]
        results: List[str] = [""] * len(texts)
        if not texts:
            return results

        glossary = self._glossary_block(series_md)
        prev_tail: List[Tuple[str, str]] = []  # (kaynak, çeviri) bağlamı

        for start in range(0, len(texts), BATCH_SIZE):
            chunk = texts[start:start + BATCH_SIZE]

            context = ""
            if prev_tail:
                ctx_lines = "\n".join(f"{src} -> {tr}" for src, tr in prev_tail)
                context = (
                    "Previous bubbles already translated (context, do NOT re-output):\n"
                    + ctx_lines + "\n\n"
                )

            numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(chunk))
            prompt = (
                glossary + context + BATCH_INSTRUCTIONS
                + "\n\n" + numbered + "\n\nTurkish translations:"
            )

            raw = self._complete(prompt)
            parsed = _parse_numbered(raw, expected=len(chunk))
            for i in range(len(chunk)):
                tr = (parsed.get(i + 1) or "").strip()
                if tr:
                    results[start + i] = tr

            # Eksik kalanları tek tek tamamla
            for i in range(len(chunk)):
                gi = start + i
                if texts[gi] and not results[gi]:
                    print(f"  Toplu çeviride eksik kaldı, tekil deneniyor: #{gi + 1}")
                    results[gi] = self.translate_single(texts[gi], series_md) or texts[gi]

            tail = [(texts[start + i], results[start + i])
                    for i in range(len(chunk)) if results[start + i]]
            prev_tail = tail[-CONTEXT_TAIL:]

        return results

    # ── Seri notları (.md) güncelleme ─────────────────────────
    def update_series_md(self, old_md: str, pairs: Sequence[Tuple[str, str]],
                         series_name: str) -> Optional[str]:
        """Bölüm çevirisinden sonra seri .md dosyasını günceller.

        Yapısal görev olduğundan önce Gemini/OpenAI denenir; translategemma
        salt çeviri modeli olduğu için en sona bırakılır. Hepsi başarısızsa
        None döner (eski dosya korunur).
        """
        if not pairs:
            return None

        pair_lines = "\n".join(f"- {src} => {tr}" for src, tr in pairs)
        old_block = (old_md or "").strip() or "(henüz yok)"
        prompt = (
            f'You maintain the Turkish translation notes for the comic series "{series_name}".\n\n'
            "CURRENT NOTES (markdown):\n-----\n" + old_block + "\n-----\n\n"
            "NEW CHAPTER TRANSLATIONS (English => Turkish):\n" + pair_lines + "\n\n"
            "Update the notes so future chapters stay consistent:\n"
            "- '## Karakterler': character names (how they are written in Turkish) "
            "and each character's speech style.\n"
            "- '## Terimler Sözlüğü': recurring terms, places, attacks, honorifics "
            "as 'English => Türkçe' bullet list.\n"
            "- '## Üslup Notları': overall tone decisions.\n"
            "- '## Bölüm Özetleri': append a 1-2 sentence summary for this chapter.\n"
            "Keep existing correct entries, merge new ones, remove duplicates.\n"
            "Write the notes themselves in Turkish.\n"
            "Output ONLY the full updated markdown document, nothing else."
        )
        system = "You are a meticulous translation-notes editor. Output markdown only."

        for name, fn in (
            ("gemini", lambda: self._call_gemini(system + "\n\n" + prompt)),
            ("openai", lambda: self._call_openai(prompt, system)),
            ("ollama", lambda: self._call_ollama(system + "\n\n" + prompt)),
        ):
            try:
                out = (fn() or "").strip()
                out = re.sub(r"^```(?:markdown)?\s*|\s*```$", "", out).strip()
                # Asgari sağlamlık kontrolü: markdown başlığı içersin, boş olmasın
                if "#" in out and len(out) > 40:
                    return out
                print(f"  MD güncellemesi geçersiz çıktı ({name}), sonraki deneniyor")
            except Exception as e:
                print(f"  MD güncelleme hatası ({name}): {e}")
        return None
