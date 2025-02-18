from typing import Dict, List, Optional
import re
import unicodedata
from arabic_reshaper import ArabicReshaper
from bidi.algorithm import get_display

class UrduTransliterator:
    def __init__(self, word_lexicon: Optional[Dict[str, str]] = None):
        # Initialize with character mappings from the paper
        self.consonant_map = {
            'ا': '',     # Initial alif is not transliterated
            'ب': 'b',
            'پ': 'p', 
            'ت': 't',
            'ٹ': 'T',
            'ث': 's2',
            'ج': 'j',
            'چ': 'c',
            'ح': 'h2',
            'خ': 'x',
            'د': 'd',
            'ڈ': 'D',
            'ذ': 'z2',
            'ر': 'r',
            'ڑ': 'R',
            'ز': 'z',
            'ژ': 'z',
            'س': 's',
            'ش': 'S',
            'ص': 's3',  
            'ض': 'z3',
            'ط': 't2',
            'ظ': 'z4',
            'ع': 'a2',
            'غ': 'G',
            'ف': 'f',
            'ق': 'q',
            'ک': 'k',
            'گ': 'g',
            'ل': 'l',
            'م': 'm',
            'ن': 'n',
            'ں': 'N',
            'و': 'v',
            'ہ': 'h',
            'ھ': 'H',
            'ء': "'",
            'ی': 'y',
            'ے': 'e'
        }

        # Vowel mappings with diacritics
        self.vowel_map = {
            'َ': 'a',    # Zabar
            'ِ': 'i',    # Zer
            'ُ': 'u',    # Pesh
            'آ': 'A',    # Alif madd
            'ٰ': 'A',    # Alif maqsurah
            'ٗ': 'U',    # Inverted Pesh
            'ً': 'an',   # Double Zabar
            'ٍ': 'in',   # Double Zer
            'ٌ': 'un',   # Double Pesh
        }

        # Special combinations
        self.special_combinations = {
            'آ': 'A',
            'او': 'o',
            'ای': 'e',
            'یا': 'yA',
            'ئ': "'",
            'ؤ': "'",
            'وا': 'vA',
        }
        
        self.word_lexicon = word_lexicon or {}

    def normalize(self, text: str) -> str:
        """Normalize the text by converting decomposed forms to composed forms"""
        return unicodedata.normalize('NFKC', text)

    def transliterate_word(self, word: str) -> str:
        """
        Transliterate a single word from Urdu to Roman script.
        Handles right-to-left script and proper vowel placement.
        """
        # Normalize the input word
        word = self.normalize(word)
        
        # Convert to list and reverse since we're processing RTL text
        chars = list(word)[::-1]
        result = []
        i = 0
        
        while i < len(chars):
            current_char = chars[i]
            
            # Skip if empty character
            if not current_char.strip():
                i += 1
                continue
                
            # Check for special combinations first
            if i < len(chars) - 1:
                two_chars = current_char + chars[i + 1]
                if two_chars in self.special_combinations:
                    result.append(self.special_combinations[two_chars])
                    i += 2
                    continue

            # Handle consonants
            if current_char in self.consonant_map:
                # Get the base consonant transliteration
                trans = self.consonant_map[current_char]
                
                # Handle initial alif specially
                if current_char == 'ا' and i == len(chars) - 1:  # Remember we reversed the string
                    if i == 0 or chars[i-1] not in self.vowel_map:
                        i += 1
                        continue
                        
                # Look ahead for vowel marks
                if i + 1 < len(chars) and chars[i + 1] in self.vowel_map:
                    vowel = self.vowel_map[chars[i + 1]]
                    trans += vowel
                    i += 2
                else:
                    # Add implicit 'a' if not word-final
                    if i != 0:  # Not the last letter of original word
                        trans += 'a'
                    i += 1
                    
                result.append(trans)
                continue

            # Handle standalone vowels
            if current_char in self.vowel_map and (i == 0 or chars[i-1] not in self.consonant_map):
                result.append(self.vowel_map[current_char])
            
            i += 1

        return ''.join(result)

    def transliterate(self, text: str) -> str:
        """Main transliteration function implementing the full pipeline"""
        # Split into words
        words = text.split()
        
        # Process each word
        transliterated_words = [self.transliterate_word(w) for w in words]
        
        # Join with spaces
        return ' '.join(transliterated_words)

# Example usage
if __name__ == "__main__":
    # Initialize transliterator
    transliterator = UrduTransliterator()
    
    # Example text
    text = "مسلمانوں"  # qalam (pen)
    
    # Transliterate
    result = transliterator.transliterate(text[::-1])
    print(f"{text} -> {result}")