"""Rule-based normalization for extracted entity names."""

import re

# Standardize legal/business suffix variants to a canonical short form.
# Each pattern is anchored to the end of the string (``(?=\s*$)``) so a rule
# only rewrites a *trailing* legal suffix, never a mid-name word — otherwise
# 'Limited Brands' would become 'Ltd Brands'.
# Order matters: longer/more-specific patterns first so they win over
# their substring forms (e.g., 'Limited Liability Company' before 'LLC').
_SUFFIX_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r'\bLimited\s+Liability\s+Co(?:mpany)?(?=\s*$)', re.IGNORECASE), 'LLC'),
    (re.compile(r'\bL\.?\s*L\.?\s*C\.?(?=\s*$)', re.IGNORECASE), 'LLC'),
    (re.compile(r'\bIncorporated(?=\s*$)', re.IGNORECASE), 'Inc'),
    (re.compile(r'\bInc\.?(?=\s*$)', re.IGNORECASE), 'Inc'),
    (re.compile(r'\bCorporation(?=\s*$)', re.IGNORECASE), 'Corp'),
    (re.compile(r'\bCorp\.?(?=\s*$)', re.IGNORECASE), 'Corp'),
    (re.compile(r'\bLimited(?=\s*$)', re.IGNORECASE), 'Ltd'),
    (re.compile(r'\bLtd\.?(?=\s*$)', re.IGNORECASE), 'Ltd'),
    (re.compile(r'\bCompany(?=\s*$)', re.IGNORECASE), 'Co'),
    (re.compile(r'\bCo\.?(?=\s*$)', re.IGNORECASE), 'Co'),
)

_PUNCT_STRIP = re.compile(r'^[\s\.,;:!?\-_()/\\"\']+|[\s\.,;:!?\-_()/\\"\']+$')
_WHITESPACE = re.compile(r'\s+')


class EntityNameNormalizer:
    """Normalize extracted entity names via a deterministic rule pipeline.

    Rule pipeline (applied in order):
        1. Trim whitespace and surrounding punctuation.
        2. Collapse internal whitespace.
        3. Standardize legal-entity suffixes (LLC, Inc, Corp, Ltd, Co).
        4. Title-case tokens, while preserving short all-caps acronyms.
    """

    @staticmethod
    def _strip_outer_punct(text: str) -> str:
        """Remove leading/trailing whitespace and punctuation."""
        return _PUNCT_STRIP.sub('', text)

    @staticmethod
    def _collapse_whitespace(text: str) -> str:
        """Collapse runs of whitespace into a single space."""
        return _WHITESPACE.sub(' ', text).strip()

    @staticmethod
    def _apply_suffix_rules(text: str) -> str:
        """Standardize known legal/business suffix variants."""
        for pattern, replacement in _SUFFIX_RULES:
            text = pattern.sub(replacement, text)
        return text

    @staticmethod
    def _smart_title_case(text: str) -> str:
        """Title-case tokens, preserving existing internal capitalization.

        A token that already contains an uppercase letter is left untouched.
        Purely lower-case tokens have their first character capitalized.
        """
        tokens = text.split(' ')
        out: list[str] = []
        for token in tokens:
            if not token:
                continue
            if any(c.isupper() for c in token):
                out.append(token)
            else:
                out.append(token[:1].upper() + token[1:])
        return ' '.join(out)

    def _rule_clean(self, text: str) -> str:
        """Apply the rule pipeline to a raw string."""
        text = self._strip_outer_punct(text)
        text = self._collapse_whitespace(text)
        text = self._apply_suffix_rules(text)
        text = self._collapse_whitespace(text)
        text = self._smart_title_case(text)
        return text.strip()

    def normalize(self, text: str) -> str:
        """Normalize a single extracted name.

        Args:
            text: Raw extracted span.

        Returns:
            Normalized entity name (possibly empty).
        """
        if not text or not text.strip():
            return ''
        return self._rule_clean(text)

    def normalize_batch(self, texts: list[str]) -> list[str]:
        """Normalize a batch of extracted names.

        Args:
            texts: List of raw extracted spans.

        Returns:
            List of normalized names (one per input).
        """
        return [self.normalize(t) for t in texts]


__all__ = ['EntityNameNormalizer']
