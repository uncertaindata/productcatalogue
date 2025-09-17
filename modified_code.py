# // Copyright (c) 2025 Sumit Jain
# // Author: Sumit Jain <sumitjain3033@gmail.com>
# // For evaluation only. NOT for commercial, production, or professional use.
# // All rights reserved. Redistribution or reuse of any part of this code
# // for company internal or external projects is prohibited without prior
# // written consent from the author.

import re
from typing import Dict, Optional
import pandas as pd
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import re
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class VariantAxes:
    """Structured representation of variant axes"""
    config: Dict[str, Any] = None
    size: Dict[str, Any] = None
    silicon: Dict[str, Any] = None
    region: Dict[str, Any] = None
    carrier: Dict[str, Any] = None
    packaging: Dict[str, Any] = None


@dataclass
class ProductGroup:
    """Represents a product family"""
    group_id: str
    brand: str
    family: str
    generation: Optional[str]
    base_specs: Dict[str, Any]
    variant_count: int = 0
    product_count: int = 0


@dataclass
class Variant:
    """Represents a specific product configuration"""
    variant_id: str
    group_id: str
    axes: Dict[str, Any]
    product_count: int = 0


@dataclass
class Assignment:
    """Maps a product to its variant with confidence"""
    product_id: str
    group_id: str
    variant_id: str
    confidence: float
    evidence: List[str]



 
import json
import re

# def parse_openai_json(content: str) -> dict:
#     """
#     Extract and parse a JSON object from an OpenAI content string
#     that may be wrapped with ```json ... ``` fencing.
#     """
#     # Strip code fences if present
#     match = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
#     if match:
#         cleaned = match.group(1)
#     else:
#         cleaned = content.strip()

#     # Parse JSON
#     return json.loads(cleaned)

# df["parsed"] = df["normalized"].apply(parse_openai_json)

import json
import re
import pandas as pd

def parse_openai_json(content: str, row_idx=None) -> dict:
    """
    Extract and parse a JSON object from an OpenAI content string
    that may be wrapped with ```json ... ``` fencing.
    If parsing fails, return an empty dict and log debug info.
    """
    try:
        if not isinstance(content, str):
            raise ValueError(f"Expected string, got {type(content)}")

        # Try to strip code fences if present
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
        cleaned = match.group(1) if match else content.strip()

        # Parse JSON
        return json.loads(cleaned)

    except Exception as e:
        print("\n[JSON Parse Error]")
        print(f"Row index: {row_idx}")
        print(f"Raw content: {repr(content)[:500]}")  # truncate long strings
        print(f"Error: {e}\n")
        return {}

# # Apply safely with row index for debugging
# df["parsed"] = [
#     parse_openai_json(val, idx) for idx, val in df["normalized"].items()
# ]




class ProductHierarchyClassifier:
    """Main classifier for building product hierarchy"""
    
    def __init__(self):
        self.product_groups: Dict[str, ProductGroup] = {}
        self.variants: Dict[str, Variant] = {}
        self.assignments: List[Assignment] = []
        self.stats = {
            'total_products': 0,
            'products_assigned': 0,
            'products_unassigned': 0
        }

        self.known_brands = [
            "apple", "dell", "hp", "hewlett-packard", "lenovo", "asus",
            "samsung", "lg", "sony", "acer", "microsoft", "toshiba",
            "msi", "huawei", "xiaomi", "google", "razer", "alienware"
        ]

                # Words we want to strip from models
        self.noise_words = {"laptop", "notebook", "series", "ultrabook"}

        self.BASE_SPECS_MAP = {
        'Laptop': {'form_factor': 'laptop'},
        'TV': {'display_type': lambda row: self.extract_display_type(row)},
        # Add more categories here if needed
        }
    def process_dataset(self, df: pd.DataFrame) -> None:
        """Main processing pipeline"""
        logger.info(f"Processing {len(df)} products...")
        
        # Step 1: Parse and clean data
        df = self.parse_product_data(df)
        
        # Step 2: Extract variant axes for each product
        df['axes'] = df.apply(self.extract_variant_axes, axis=1)
        
        # Step 3: Create product groups
        self.create_product_groups(df)
        
        # Step 4: Create variants and assign products
        self.create_variants_and_assign(df)
        
        # Step 5: Calculate statistics
        self.calculate_statistics(df)
        
        logger.info("Processing complete!")
    
    def parse_product_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse JSON details and extract key fields"""
        logger.info("Parsing product data...")
        
        def safe_json_parse(x):
            if pd.isna(x):
                return {}
            try:
                return json.loads(x) if isinstance(x, str) else x
            except:
                return {}
        
        df['details_parsed'] = df['details'].apply(safe_json_parse)
        
        # Extract commonly needed fields
        df['brand_clean'] = df.apply(self.extract_brand, axis=1)
        df['model_clean'] = df.apply(self.extract_model, axis=1)
        
        return df
    
    def extract_variant_axes(self, row: pd.Series) -> VariantAxes:
        """Extract normalized variant axes from a product"""
        axes = VariantAxes()
        
        # Extract config axis (RAM, storage, color)
        axes.config = self.extract_config_axis(row)
        
        # Extract size axis (screen size, dimensions)
        axes.size = self.extract_size_axis(row)
        
        # Extract silicon axis (CPU, GPU)
        axes.silicon = self.extract_silicon_axis(row)
        
        # Extract packaging axis (condition, bundle status)
        axes.packaging = self.extract_packaging_axis(row)
        
        return axes
    
    def extract_config_axis(self, row: pd.Series) -> Dict[str, Any]:
        """Extract configuration attributes (RAM, storage, color)"""
        config = {}
        details = row.get('details_parsed', {})
        
        # Extract RAM
        # ram = self.extract_ram(row['name'], details)
        ram = self.extract_ram(row, details)
        if ram:
            config['ram_gb'] = ram
        
        # Extract storage
        # storage = self.extract_storage(row['name'], details)
        storage = self.extract_storage(row, details)
        if storage:
            config['storage_gb'] = storage
        
        # Extract color
        # color = self.extract_color(row['name'], details)
        color = self.extract_color(row, details)
        if color:
            config['color'] = color
        
        return config if config else None
    
    def extract_size_axis(self, row: pd.Series) -> Dict[str, Any]:
        """Extract size attributes (screen size, physical dimensions)"""
        size = {}
        details = row.get('details_parsed', {})
        
        # Extract screen size for laptops and TVs
        # screen_size = self.extract_screen_size(row['name'], details)
        screen_size = self.extract_screen_size(row , details)
        if screen_size:
            size['screen_inches'] = screen_size
        
        return size if size else None
    
    # def extract_silicon_axis(self, row: pd.Series) -> Dict[str, Any]:
    #     """Extract silicon/processor attributes"""
    #     silicon = {}
    #     details = row.get('details_parsed', {})
        
    #     # Extract CPU
    #     cpu = self.extract_cpu(row['name'], details)
    #     if cpu:
    #         silicon['cpu'] = cpu
        
    #     # Extract GPU if present
    #     gpu = self.extract_gpu(row['name'], details)
    #     if gpu:
    #         silicon['gpu'] = gpu
        
    #     return silicon if silicon else None

    def extract_silicon_axis(self, row: pd.Series) -> Optional[Dict[str, Any]]:
        """Extract silicon/processor attributes (CPU, GPU)"""
        silicon = {}
        details = row.get('details_parsed', {})
    
        # 1. Extract CPU
        cpu = self.extract_cpu(row, details)
        if cpu:
            silicon['cpu'] = cpu
    
        # 2. Extract GPU
        gpu = self.extract_gpu(row, details)
        if gpu:
            silicon['gpu'] = gpu
    
        return silicon if silicon else None 
    
    # def extract_packaging_axis(self, row: pd.Series) -> Dict[str, Any]:
    #     """Extract packaging attributes (condition, bundle)"""
    #     return {
    #         'condition': 'new',  # Default to new unless specified otherwise
    #         'bundle': self.is_bundle(row['name'])
    #     }
    # def is_bundle(self, name: str) -> bool:
    #     """Check if product is a bundle"""
    #     bundle_keywords = ['bundle', 'kit', 'combo', 'pack', 'with', '+', '&', 'includes']
    #     name_lower = name.lower()
    #     return any(keyword in name_lower for keyword in bundle_keywords)

        
    def extract_packaging_axis(self, row: pd.Series) -> Optional[Dict[str, Any]]:
        """
        Extract packaging attributes: condition and bundle.
        Priority:
          1. parsed dict
          2. details_parsed
          3. product name
        """
        packaging = {}
    
        # 1. Try parsed dict
        parsed = row.get("parsed", {})
        if isinstance(parsed, dict):
            condition = parsed.get("packaging_condition", "")
            if condition:
                packaging['condition'] = self.normalize_condition(condition)
    
        # 2. Try details_parsed
        if 'condition' not in packaging:
            details = row.get("details_parsed", {})
            condition = details.get("packaging_condition", "")
            if condition:
                packaging['condition'] = self.normalize_condition(condition)
    
        # 3. Default to 'new' if still not found
        if 'condition' not in packaging or not packaging['condition']:
            packaging['condition'] = 'new'
    
        # Determine if it's a bundle
        name = str(row.get("name", ""))
        packaging['bundle'] = self.is_bundle(name)
    
        return packaging if packaging else None



    def normalize_condition(self, condition: str) -> str:
        """
        Normalize packaging condition to standard tokens:
          - new, refurbished, like new, used, open box
        Priority:
          1. like new (exact phrase)
          2. refurbished / restored
          3. new / sealed
          4. open box
          5. used
          6. default fallback
        """
        condition = str(condition).lower().strip()
    
        if 'like new' in condition:
            return 'refurbished_like_new'
        elif any(k in condition for k in ['refurb', 'restored']):
            return 'refurbished'
        elif any(k in condition for k in ['new', 'sealed']):
            return 'new'
        elif 'open box' in condition:
            return 'open_box'
        elif 'used' in condition:
            return 'used'
        else:
            return 'new'  # default fallback
    
    def is_bundle(self, name: str) -> bool:
        """
        Check if product is a bundle.
        Looks for keywords like bundle, kit, combo, pack, +, &, includes
        """
        bundle_keywords = ['bundle', 'kit', 'combo', 'pack', 'with', '+', '&', 'includes']
        name_lower = str(name).lower()
        return any(keyword in name_lower for keyword in bundle_keywords)

    
    # def extract_screen_size(self, name: str, details: Dict) -> Optional[float]:
    #     """Extract screen size in inches"""
    #     # Patterns like "13.6 inch", "15.6"", "65 inch", "55""
    #     patterns = [
    #         r'(\d+\.?\d*)\s*(?:inch|")',
    #         r'(\d+\.?\d*)-inch',
    #         r'(\d+\.?\d*)"\s',
    #     ]
        
    #     for pattern in patterns:
    #         match = re.search(pattern, name, re.IGNORECASE)
    #         if match:
    #             return float(match.group(1))
        
    #     return None
        
    def extract_screen_size(self, row: pd.Series, details: Dict) -> Optional[float]:
        """
        Extract size attributes (screen size) for electronics.
        Priority:
          1. parsed dict
          2. direct 'screen_size' field
          3. name
          4. variant_axes
        Returns a dictionary like {'screen_inches': 13.3} or None if not found.
        """
        
        def parse_screen_size(text: str) -> Optional[float]:
            """
            Extract screen size in inches from text.
            Handles patterns like:
              - 13.3 inch
              - 15.6"
              - 65 inch
              - 15-inch
              - 15in
            """
            if not text:
                return None
        
            text = str(text).lower()
        
            # Regex patterns
            patterns = [
                r'(\d+(?:\.\d+)?)\s*(?:inch|inches|")',  # 13.3 inch, 15"
                r'(\d+(?:\.\d+)?)-inch',                 # 15-inch
                r'(\d+(?:\.\d+)?)\s*in\b',               # 15in
            ]
        
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    return float(match.group(1))
        
            return None

    
        # 1. Try parsed dict first
        parsed = row.get("parsed", {})
        if isinstance(parsed, dict):
            parsed_screen = parsed.get("screen_size", "")
            # print(f'found screen in parsed openai', parsed_screen)
            screen_inches = parse_screen_size(parsed_screen)
            # print(f'found screen in parsed openai', parsed_screen)

            if screen_inches:
                return screen_inches
    
        # 2. Try direct screen_size field
        # direct_screen = row.get("screen_size", "")
        screen_inches = parse_screen_size(details)
        if screen_inches:
            return screen_inches
    
        # 3. Try product name
        name = str(row.get("name", ""))
        screen_inches = parse_screen_size(name)
        if screen_inches:
            return screen_inches
    
        # 4. Try variant_axes
        variant_axes = row.get("variant_axes", {})
        if isinstance(variant_axes, dict):
            variant_size = variant_axes.get("size", "")
            screen_inches = parse_screen_size(str(variant_size))
            if screen_inches:
                return screen_inches
    
        return None

    # Extraction helper methods
    # def extract_brand(self, row: pd.Series) -> str:
    #     """Extract and normalize brand name"""
    #     # Implementation needed: extract from name, details, or brand column
    #     # Normalize variations (HP vs Hewlett-Packard, etc.)
    #     brand = row.get('brand', '').lower().strip()
    #     if not brand:
    #         # Try to extract from name or details
    #         name = row.get('name', '').lower()
    #         for known_brand in ['apple', 'dell', 'hp', 'lenovo', 'asus', 'samsung', 'lg', 'sony']:
    #             if known_brand in name:
    #                 return known_brand
    #     return brand
        
        
    def extract_brand(self, row: pd.Series) -> str:
        parsed = row.get("parsed", {})
        brand = ""

        if isinstance(parsed, dict):
            brand = parsed.get("brand", "")
            if brand:
                brand = brand.lower().strip()

        # If brand missing, try fallback
        if not brand:
            name = str(row.get("name", "")).lower()
            details = str(row.get("details", "")).lower()

            for known_brand in self.known_brands:
                if known_brand in name or known_brand in details:
                    brand = known_brand
                    break

        return brand
    # def extract_model(self, row: pd.Series) -> str:
    #     """Extract model identifier"""
    #     # Implementation needed
    #     return row.get('model', '').strip()
    # def extract_model(self, row: pd.Series) -> str:
    #     """Extract model identifier"""
    #     parsed = row.get("parsed", {})
    #     model = ""

    #     if isinstance(parsed, dict):
    #         model = parsed.get("model", "")
    #         if model:
    #             return model.strip()

    #     # Fallback: look inside name/details for model-like patterns
    #     text = " ".join([str(row.get("name", "")), str(row.get("details", ""))])

    #     # Common model pattern: letters + digits + optional slash/dash
    #     match = re.search(r"\b([A-Z0-9-]{3,}[A-Z0-9/-]*)\b", text, re.IGNORECASE)
    #     if match:
    #         return match.group(1).strip()

    #     return ""

    def normalize_model(self, model: str) -> str:
        """Standardize model string"""
        if not model:
            return ""

        model = model.strip()

        # Remove brand/family noise words
        parts = [w for w in model.split() if w.lower() not in self.noise_words]
        model = " ".join(parts)

        # Uppercase alphanumeric tokens like md760ll/a -> MD760LL/A
        model = re.sub(r"\b([a-z0-9/-]+)\b", lambda m: m.group(1).upper(), model)

        # Insert space if needed in things like XPS13 -> XPS 13
        model = re.sub(r"([A-Z]+)(\d+)", r"\1 \2", model)

        # Clean repeated spaces
        model = re.sub(r"\s+", " ", model).strip()

        return model

    def extract_model(self, row: pd.Series) -> str:
        """Extract and normalize model identifier"""
        parsed = row.get("parsed", {})
        model = ""

        if isinstance(parsed, dict):
            model = parsed.get("model", "")
            if model:
                return self.normalize_model(model)

        # Fallback: look inside name/details for model-like patterns
        text = " ".join([str(row.get("name", "")), str(row.get("details", ""))])

        match = re.search(r"\b([A-Z0-9-]{3,}[A-Z0-9/-]*)\b", text, re.IGNORECASE)
        if match:
            return self.normalize_model(match.group(1))

        return ""
        
    # def extract_ram(self, name: str, details: Dict) -> Optional[int]:
    #     """Extract RAM in GB"""
    #     # Look for patterns like "8GB RAM", "16 GB Memory"
    #     # Check in specifications JSON
    #     # Normalize to integer GB value
        
    #     # Simple regex pattern - extend as needed
    #     ram_pattern = r'(\d+)\s*GB\s*(?:RAM|Memory|DDR)'
    #     match = re.search(ram_pattern, name, re.IGNORECASE)
    #     if match:
    #         return int(match.group(1))
        
    #     # Check in specifications
    #     specs = details.get('specifications', {})
    #     if isinstance(specs, dict):
    #         for key, value in specs.items():
    #             if 'ram' in key.lower() or 'memory' in key.lower():
    #                 # Parse value to extract number
    #                 match = re.search(r'(\d+)', str(value))
    #                 if match:
    #                     return int(match.group(1))
        
    #     return None

    
    def extract_ram(self, row: pd.Series, details: Dict) -> Optional[int]:
        """Extract RAM in GB from parsed, specifications, or free-text fields"""
    
        def parse_ram(text: str) -> Optional[int]:
            if not text:
                return None
    
            # Look for GB
            match = re.search(r'(\d+)\s*GB', text, re.IGNORECASE)
            if match:
                return int(match.group(1))
    
            # Look for MB (convert to GB)
            match = re.search(r'(\d+)\s*MB', text, re.IGNORECASE)
            if match:
                mb_val = int(match.group(1))
                return max(1, mb_val // 1024)  # convert MB → GB
    
            return None
    
        # 1. Try parsed dict first
        parsed = row.get("parsed", {})
        if isinstance(parsed, dict):
            ram_val = parse_ram(parsed.get("ram", ""))
            if ram_val:
                return ram_val
    
        specs = details.get("specifications", {}) if isinstance(details, dict) else {}
        if isinstance(specs, dict):
            for key, value in specs.items():
                if any(word in key.lower() for word in ["ram", "memory", "ddr"]):
                    ram_val = parse_ram(str(value))
                    if ram_val:
                        return ram_val
    
        # 3. Fallback: scan free-text in name/details
        text_fields = [
            str(row.get("name", "")),
            str(details) if isinstance(details, dict) else str(details)
        ]
        combined_text = " ".join(text_fields)
        ram_val = parse_ram(combined_text)
        if ram_val:
            return ram_val
    
        return None

    # def extract_storage(self, name: str, details: Dict) -> Optional[int]:
    #     """Extract storage in GB"""
    #     # Look for patterns like "256GB", "1TB", "512 GB SSD"
    #     # Convert TB to GB (1TB = 1024GB)
        
    #     # Check for TB first
    #     tb_pattern = r'(\d+)\s*TB'
    #     match = re.search(tb_pattern, name, re.IGNORECASE)
    #     if match:
    #         return int(match.group(1)) * 1024
        
    #     # Then check for GB
    #     gb_pattern = r'(\d+)\s*GB\s*(?:SSD|HDD|Storage)?'
    #     match = re.search(gb_pattern, name, re.IGNORECASE)
    #     if match:
    #         return int(match.group(1))
        
    #     return None


    # def extract_storage(self, row: pd.Series , details: Dict) -> Optional[int]:
    #     name = row.get("name", "")
    #     """Extract storage in GB"""
    #     # Look for patterns like "256GB", "1TB", "512 GB SSD"
    #     # Convert TB to GB (1TB = 1024GB)
        
    #     # Check for TB first
    #     tb_pattern = r'(\d+)\s*TB'
    #     match = re.search(tb_pattern, name, re.IGNORECASE)
    #     if match:
    #         return int(match.group(1)) * 1024
        
    #     # Then check for GB
    #     gb_pattern = r'(\d+)\s*GB\s*(?:SSD|HDD|Storage)?'
    #     match = re.search(gb_pattern, name, re.IGNORECASE)
    #     if match:
    #         return int(match.group(1))
        
    #     return None

    def extract_storage(self, row: pd.Series, details: Dict) -> Optional[int]:
        """Extract and normalize storage size in GB"""
        parsed = row.get("parsed", {})
        
        # Step 1: Check parsed column
        if isinstance(parsed, dict):
            storage_val = parsed.get("storage", "")
            if storage_val:
                tb_match = re.search(r'(\d+)\s*TB', storage_val, re.IGNORECASE)
                if tb_match:
                    return int(tb_match.group(1)) * 1024
                
                gb_match = re.search(r'(\d+)\s*GB', storage_val, re.IGNORECASE)
                if gb_match:
                    return int(gb_match.group(1))
    
        # Step 2: Fallback — look inside name and details
        text = " ".join([str(row.get("name", "")), str(details)])
        
        tb_match = re.search(r'(\d+)\s*TB', text, re.IGNORECASE)
        if tb_match:
            return int(tb_match.group(1)) * 1024
        
        gb_match = re.search(r'(\d+)\s*GB\s*(?:SSD|HDD|Storage)?', text, re.IGNORECASE)
        if gb_match:
            return int(gb_match.group(1))
    
        return None

    # def extract_color(self, name: str, details: Dict) -> Optional[str]:

    # def extract_color(self, row: pd.Series, details: Dict) -> Optional[int]:
    #     name = row.get("name", "")
    #     """Extract and normalize color"""
    #     colors = {
    #         'silver': ['silver', 'platinum'],
    #         'space_gray': ['space gray', 'space grey', 'gray', 'grey'],
    #         'gold': ['gold', 'champagne'],
    #         'black': ['black', 'carbon'],
    #         'white': ['white', 'pearl'],
    #         'blue': ['blue', 'navy', 'midnight'],
    #         'red': ['red', 'rose', 'pink']
    #     }
        
    #     name_lower = name.lower()
    #     for normalized, variations in colors.items():
    #         for variation in variations:
    #             if variation in name_lower:
    #                 return normalized
        
    #     # Check in details
    #     color_field = details.get('color', '')
    #     if color_field:
    #         color_lower = color_field.lower()
    #         for normalized, variations in colors.items():
    #             for variation in variations:
    #                 if variation in color_lower:
    #                     return normalized
        
    #     return None

    def extract_color(self, row: pd.Series, details: Dict) -> Optional[str]:
        """Extract and normalize color, priority:
           1. parsed dict
           2. name
           3. details
        """
        
            
        # def normalize_color(raw_color: str) -> Optional[str]:
        #     raw_color_lower = raw_color.lower()
        #     for normalized, variants in colors.items():
        #         if any(v in raw_color_lower for v in variants):
        #             return normalized
        #     return None


        def normalize_color(raw_color: str) -> Optional[str]:
            """
            Normalize color string for electronics (laptops, TVs, monitors) 
            to a predefined set of standard colors.
            """
            if not raw_color:
                return None
        
            raw_color_lower = raw_color.lower()
        
            colors = {
                'silver': ['silver', 'platinum', 'metallic silver', 'aluminum', 'light silver'],
                'space_gray': ['space gray', 'space grey', 'gray', 'grey', 'charcoal', 'graphite', 'gunmetal', 'dark gray'],
                'gold': ['gold', 'champagne', 'rose gold', 'pink gold', 'light gold'],
                'black': ['black', 'jet black', 'matte black', 'onyx', 'carbon', 'midnight black'],
                'white': ['white', 'pearl', 'ivory', 'snow', 'moonlight'],
                'blue': ['blue', 'navy', 'midnight', 'royal blue', 'cobalt', 'sky blue'],
                'red': ['red', 'burgundy', 'crimson', 'ruby'],
                'green': ['green', 'olive', 'sage', 'forest green', 'mint'], 
                'brown': ['brown', 'chocolate', 'mocha', 'bronze', 'copper'],
                'grey': ['grey', 'dark grey', 'light grey'],
                'teal': ['teal', 'aqua', 'turquoise', 'cyan'], 
                'gold_black_combo': ['gold/black', 'two-tone gold', 'black/gold'],
                'silver_black_combo': ['silver/black', 'two-tone silver', 'black/silver'],
                'white_black_combo': ['white/black', 'two-tone white', 'black/white']
            }
        
            for normalized, variants in colors.items():
                if any(variant in raw_color_lower for variant in variants):
                    return normalized
        
            return None

        # 1. Try parsed dict first
        parsed = details.get("parsed", {})
        if isinstance(parsed, dict):
            parsed_color = parsed.get("color", "")
            if parsed_color:
                normalized = normalize_color(parsed_color)
                if normalized:
                    return normalized
    
        # 2. Try name field
        name = str(row.get("name", ""))
        normalized = normalize_color(name)
        if normalized:
            return normalized
    
        # 3. Try details field
        color_field = str(details.get("color", ""))
        normalized = normalize_color(color_field)
        if normalized:
            return normalized
    
        # 4. Could not find
        return None


    # def extract_cpu(self, name: str, details: Dict) -> Optional[str]:
    #     """Extract and normalize CPU model"""
    #     # Look for Intel, AMD, Apple Silicon models
    #     # Normalize to standard tokens
        
    #     cpu_patterns = {
    #         'apple_m1': r'M1(?:\s+Pro|\s+Max)?',
    #         'apple_m2': r'M2(?:\s+Pro|\s+Max)?',
    #         'apple_m3': r'M3(?:\s+Pro|\s+Max)?',
    #         'intel_i5': r'(?:Intel\s+)?(?:Core\s+)?i5(?:-\d+)?',
    #         'intel_i7': r'(?:Intel\s+)?(?:Core\s+)?i7(?:-\d+)?',
    #         'intel_i9': r'(?:Intel\s+)?(?:Core\s+)?i9(?:-\d+)?',
    #         'amd_ryzen5': r'(?:AMD\s+)?Ryzen\s+5',
    #         'amd_ryzen7': r'(?:AMD\s+)?Ryzen\s+7',
    #         'amd_ryzen9': r'(?:AMD\s+)?Ryzen\s+9',
    #     }
        
    #     name_and_details = name + ' ' + str(details)
    #     for cpu_token, pattern in cpu_patterns.items():
    #         if re.search(pattern, name_and_details, re.IGNORECASE):
    #             return cpu_token
        
    #     return None
    
    # def extract_gpu(self, name: str, details: Dict) -> Optional[str]:
    #     """Extract GPU model if present"""
    #     # Implementation needed
    #     return None

    def extract_cpu(self, row: pd.Series, details: Dict) -> Optional[str]:
        """
        Extract and normalize CPU model.
        Priority:
          1. parsed dict
          2. details fields
          3. product name
        """
        # cpu_patterns = {
        #     'apple_m1': r'M1(?:\s+Pro|\s+Max)?',
        #     'apple_m2': r'M2(?:\s+Pro|\s+Max)?',
        #     'apple_m3': r'M3(?:\s+Pro|\s+Max)?',
        #     'intel_i5': r'(?:Intel\s+)?(?:Core\s+)?i5(?:-\d+)?',
        #     'intel_i7': r'(?:Intel\s+)?(?:Core\s+)?i7(?:-\d+)?',
        #     'intel_i9': r'(?:Intel\s+)?(?:Core\s+)?i9(?:-\d+)?',
        #     'amd_ryzen5': r'(?:AMD\s+)?Ryzen\s+5',
        #     'amd_ryzen7': r'(?:AMD\s+)?Ryzen\s+7',
        #     'amd_ryzen9': r'(?:AMD\s+)?Ryzen\s+9',
        # }

        cpu_patterns = {
            # Apple Silicon
            'apple_m1': r'M1(?:\s*Pro|\s*Max)?',
            'apple_m2': r'M2(?:\s*Pro|\s*Max)?',
            'apple_m3': r'M3(?:\s*Pro|\s*Max)?',
        
            # Intel Core Desktop & Mobile
            'intel_i3': r'(?:Intel\s+)?(?:Core\s+)?i3(?:-\d+)?',
            'intel_i5': r'(?:Intel\s+)?(?:Core\s+)?i5(?:-\d+)?',
            'intel_i7': r'(?:Intel\s+)?(?:Core\s+)?i7(?:-\d+)?',
            'intel_i9': r'(?:Intel\s+)?(?:Core\s+)?i9(?:-\d+)?',
        
            # Intel Pentium & Celeron (entry-level)
            'intel_pentium': r'Pentium',
            'intel_celeron': r'Celeron',
        
            # Intel Xeon (workstation/server)
            'intel_xeon': r'Xeon',
        
            # AMD Ryzen Desktop & Mobile
            'amd_ryzen3': r'Ryzen\s+3',
            'amd_ryzen5': r'Ryzen\s+5',
            'amd_ryzen7': r'Ryzen\s+7',
            'amd_ryzen9': r'Ryzen\s+9',
        
            # AMD Threadripper / EPYC
            'amd_threadripper': r'Threadripper',
            'amd_epyc': r'EPYC',
        
            # Generic fallback
            'intel_generic': r'Intel',
            'amd_generic': r'AMD',
            'apple_generic': r'M\d'
        }

    
        # 1. Try parsed dict first
        parsed = row.get("parsed", {})
        if isinstance(parsed, dict):
            cpu_text = parsed.get("cpu", "")
            for token, pattern in cpu_patterns.items():
                if re.search(pattern, str(cpu_text), re.IGNORECASE):
                    return token
    
        # 2. Try details fields
        for key, value in details.items():
            if key == "parsed":
                continue
            text = str(value)
            for token, pattern in cpu_patterns.items():
                if re.search(pattern, text, re.IGNORECASE):
                    return token
    
        # 3. Try product name
        name = str(row.get("name", ""))
        for token, pattern in cpu_patterns.items():
            if re.search(pattern, name, re.IGNORECASE):
                return token
    
        return None


    def extract_gpu(self, row: pd.Series, details: Dict) -> Optional[str]:
        """
        Extract and normalize GPU model if present.
        Priority:
          1. parsed dict
          2. details fields
          3. product name
        """
        # gpu_patterns = {
        #     'nvidia_rtx_3060': r'RTX\s*3060',
        #     'nvidia_rtx_3070': r'RTX\s*3070',
        #     'nvidia_rtx_3080': r'RTX\s*3080',
        #     'nvidia_rtx_4090': r'RTX\s*4090',
        #     'amd_rx_6600': r'RX\s*6600',
        #     'amd_rx_6700': r'RX\s*6700',
        #     'amd_rx_6800': r'RX\s*6800',
        #     'amd_rx_6900': r'RX\s*6900',
        #     'intel_iris': r'Iris',
        #     'apple_gpu': r'M1\s*GPU|M2\s*GPU|M3\s*GPU',
        # }
        gpu_patterns = {
            # NVIDIA Desktop GPUs
            'nvidia_rtx_3060': r'RTX\s*3060(?:\s*Ti|\s*Super)?',
            'nvidia_rtx_3070': r'RTX\s*3070(?:\s*Ti|\s*Super)?',
            'nvidia_rtx_3080': r'RTX\s*3080(?:\s*Ti|\s*Super)?',
            'nvidia_rtx_3090': r'RTX\s*3090(?:\s*Ti|\s*Super)?',
            'nvidia_rtx_4090': r'RTX\s*4090',
            'nvidia_gtx_1660': r'GTX\s*1660(?:\s*Ti)?',
        
            # NVIDIA Laptop GPUs
            'nvidia_rtx_3060_laptop': r'RTX\s*3060\s*(Laptop|Mobile)?',
            'nvidia_rtx_3070_laptop': r'RTX\s*3070\s*(Laptop|Mobile)?',
            'nvidia_rtx_3080_laptop': r'RTX\s*3080\s*(Laptop|Mobile)?',
        
            # AMD Desktop GPUs
            'amd_rx_6600': r'RX\s*6600(?:\s*XT)?',
            'amd_rx_6700': r'RX\s*6700(?:\s*XT)?',
            'amd_rx_6800': r'RX\s*6800(?:\s*XT)?',
            'amd_rx_6900': r'RX\s*6900(?:\s*XT)?',
        
            # AMD Laptop GPUs
            'amd_rx_6800m': r'RX\s*6800M',
            'amd_rx_6700m': r'RX\s*6700M',
        
            # Intel Integrated GPUs
            'intel_iris': r'Iris(?:\s*Xe)?',
            'intel_uhd': r'UHD Graphics',
        
            # Apple GPUs
            'apple_m1_gpu': r'M1\s*GPU',
            'apple_m1_pro_gpu': r'M1\s*Pro\s*GPU',
            'apple_m1_max_gpu': r'M1\s*Max\s*GPU',
            'apple_m2_gpu': r'M2\s*GPU',
            'apple_m2_pro_gpu': r'M2\s*Pro\s*GPU',
            'apple_m2_max_gpu': r'M2\s*Max\s*GPU',
            'apple_m3_gpu': r'M3\s*GPU',
                    
            # NVIDIA 40-series Desktop GPUs
            'nvidia_rtx_4060': r'RTX\s*4060(?:\s*Ti)?',
            'nvidia_rtx_4070': r'RTX\s*4070(?:\s*Ti)?',
            'nvidia_rtx_4080': r'RTX\s*4080(?:\s*Ti)?',
            'nvidia_rtx_4090': r'RTX\s*4090',  # already included
            
            # NVIDIA 40-series Laptop GPUs
            'nvidia_rtx_4060_laptop': r'RTX\s*4060\s*(Laptop|Mobile)?',
            'nvidia_rtx_4070_laptop': r'RTX\s*4070\s*(Laptop|Mobile)?',
            'nvidia_rtx_4080_laptop': r'RTX\s*4080\s*(Laptop|Mobile)?',
            'nvidia_rtx_4090_laptop': r'RTX\s*4090\s*(Laptop|Mobile)?',
            
            # Generic fallback
            'nvidia_generic': r'GeForce',
            'amd_generic': r'Radeon',
            'intel_generic': r'Intel\s*(Iris|UHD)',
        }
        # 1. Try parsed dict first
        parsed = row.get("parsed", {})
        if isinstance(parsed, dict):
            gpu_text = parsed.get("gpu", "")
            for token, pattern in gpu_patterns.items():
                if re.search(pattern, str(gpu_text), re.IGNORECASE):
                    return token
    
        # 2. Try details fields
        for key, value in details.items():
            if key == "parsed":
                continue
            text = str(value)
            for token, pattern in gpu_patterns.items():
                if re.search(pattern, text, re.IGNORECASE):
                    return token
    
        # 3. Try product name
        name = str(row.get("name", ""))
        for token, pattern in gpu_patterns.items():
            if re.search(pattern, name, re.IGNORECASE):
                return token
    
        return None

    

    def create_product_groups(self, df: pd.DataFrame) -> None:
        """Create ProductGroups by clustering similar products"""
        logger.info("Creating product groups...")
        
        # Group by brand and model family
        # This is a simplified approach - you might want to use
        # more sophisticated clustering or ML techniques
        
        for _, row in df.iterrows():
            brand = row.get('brand_clean', 'unknown')
            model = row.get('model_clean', 'unknown')
            try:
            # Generate group ID
                group_id = self.generate_group_id(brand, model)
            except:
                print(row)
            if group_id not in self.product_groups:
                # Create new group
                group = ProductGroup(
                    group_id=group_id,
                    brand=brand,
                    family=model,
                    generation=self.extract_generation(row),
                    base_specs=self.extract_base_specs(row)
                )
                self.product_groups[group_id] = group
    
    def create_variants_and_assign(self, df: pd.DataFrame) -> None:
        """Create variants and assign products to them"""
        logger.info("Creating variants and assigning products...")
        
        for _, row in df.iterrows():
            # Get the product's group
            brand = row.get('brand_clean', 'unknown')
            model = row.get('model_clean', 'unknown')
            try:
                group_id = self.generate_group_id(brand, model)
            except:
                print(row)
            # Generate variant ID from axes
            axes = row.get('axes')
            variant_id = self.generate_variant_id(group_id, axes)
            
            # Create variant if it doesn't exist
            if variant_id not in self.variants:
                variant = Variant(
                    variant_id=variant_id,
                    group_id=group_id,
                    axes=self.axes_to_dict(axes)
                )
                self.variants[variant_id] = variant
            
            # Create assignment
            confidence, evidence = self.calculate_confidence(row)
            assignment = Assignment(
                product_id=row['product_id'],
                group_id=group_id,
                variant_id=variant_id,
                confidence=confidence,
                evidence=evidence
            )
            self.assignments.append(assignment)
            
            # Update counts
            self.variants[variant_id].product_count += 1
            self.product_groups[group_id].product_count += 1
    
    def generate_group_id(self, brand: str, model: str) -> str:
        """Generate deterministic group ID"""
        def safe_clean(x):
            if not isinstance(x, str) or not x.strip():
                print(x)
                return "unk"
            return re.sub(r'[^a-z0-9]', '_', x.lower().strip())

        
        # Normalize and combine brand and model
        brand_clean = re.sub(r'[^a-z0-9]', '_', brand.lower())
        model_clean = re.sub(r'[^a-z0-9]', '_', model.lower())
        # brand_clean = safe_clean(brand)
        # model_clean = safe_clean(model)
        return f"{brand_clean}_{model_clean}"

 


    
    def generate_variant_id(self, group_id: str, axes: VariantAxes) -> str:
        """Generate deterministic variant ID from axes"""
        variant_parts = [group_id]
        
        # Add each axis if present
        if axes.config:
            config_str = self.serialize_axis('config', axes.config)
            variant_parts.append(config_str)
        
        if axes.size:
            size_str = self.serialize_axis('size', axes.size)
            variant_parts.append(size_str)
        
        if axes.silicon:
            silicon_str = self.serialize_axis('silicon', axes.silicon)
            variant_parts.append(silicon_str)
        
        return '/'.join(variant_parts)
    
    def serialize_axis(self, axis_name: str, axis_data: Dict) -> str:
        """Serialize an axis to a string representation"""
        if not axis_data:
            return ""
        
        # Sort keys for deterministic output
        sorted_items = sorted(axis_data.items())
        values = '_'.join(str(v) for k, v in sorted_items)
        return f"{axis_name}:{values}"
    
    def axes_to_dict(self, axes: VariantAxes) -> Dict:
        """Convert VariantAxes to dictionary"""
        result = {}
        if axes.config:
            result['config'] = axes.config
        if axes.size:
            result['size'] = axes.size
        if axes.silicon:
            result['silicon'] = axes.silicon
        if axes.region:
            result['region'] = axes.region
        if axes.carrier:
            result['carrier'] = axes.carrier
        if axes.packaging:
            result['packaging'] = axes.packaging
        return result
    
    # def extract_generation(self, row: pd.Series) -> Optional[str]:
    #     """Extract product generation or year"""
    #     # Look for year patterns or generation markers
    #     name = row.get('name', '')
    #     year_match = re.search(r'20\d{2}', name)
    #     if year_match:
    #         return year_match.group(0)
    #     return None
    def extract_generation(self, row: pd.Series) -> Optional[str]:
        """Extract product generation or year"""
        name = row.get('name', '')
        
        # Check for year first
        year_match = re.search(r'20\d{2}', name)
        if year_match:
            return year_match.group(0)
        
        # Check for Intel/Apple generation patterns
        gen_match = re.search(r'\b(i\d-[0-9]{1,2}(th|st|nd|rd)|M[1-3](?:\sPro|\sMax)?)\b', name, re.IGNORECASE)
        if gen_match:
            return gen_match.group(0)
        
        return None

        
    # def extract_base_specs(self, row: pd.Series) -> Dict[str, Any]:
    #     """Extract specifications that are common to all variants in a group"""
    #     # This would identify specs that don't vary within a product family
    #     base_specs = {}
        
    #     # Example: form factor for laptops, panel type for TVs
    #     if row.get('sub_category') == 'Laptop':
    #         base_specs['form_factor'] = 'laptop'
    #     elif row.get('sub_category') == 'TV':
    #         base_specs['display_type'] = self.extract_display_type(row)
        
    #     return base_specs



    
    def extract_base_specs(self, row: pd.Series) -> Dict[str, Any]:
        """
        Extract specifications that are common to all variants in a group.
        Looks up BASE_SPECS_MAP for default axes per sub-category.
        """
        base_specs = {}
        sub_cat = row.get('sub_category')
    
        if sub_cat in self.BASE_SPECS_MAP:
            spec_map = self.BASE_SPECS_MAP[sub_cat]
            for key, val in spec_map.items():
                base_specs[key] = val(row) if callable(val) else val
    
        return base_specs

    def extract_display_type(self, row: pd.Series) -> str:
        """Extract display type for TVs"""
        name = row.get('name', '').upper()
        if 'OLED' in name:
            return 'OLED'
        elif 'QLED' in name:
            return 'QLED'
        elif 'LED' in name:
            return 'LED'
        return 'LCD'
    
    def calculate_confidence(self, row: pd.Series) -> Tuple[float, List[str]]:
        """Calculate confidence score for assignment"""
        confidence = 0.5  # Base confidence
        evidence = []
        
        # Increase confidence based on available data
        if row.get('brand_clean'):
            confidence += 0.1
            evidence.append('brand_match')
        
        if row.get('model_clean'):
            confidence += 0.1
            evidence.append('model_match')
        
        axes = row.get('axes')
        if axes:
            if axes.config:
                confidence += 0.1
                evidence.append('config_extracted')
            if axes.size:
                confidence += 0.1
                evidence.append('size_extracted')
            if axes.silicon:
                confidence += 0.1
                evidence.append('silicon_extracted')
        
        # Cap confidence at 1.0
        confidence = min(confidence, 1.0)
        
        return confidence, evidence
    
    def calculate_statistics(self, df: pd.DataFrame) -> None:
        """Calculate summary statistics"""
        self.stats['total_products'] = len(df)
        self.stats['products_assigned'] = len(self.assignments)
        self.stats['products_unassigned'] = len(df) - len(self.assignments)
        self.stats['total_groups'] = len(self.product_groups)
        self.stats['total_variants'] = len(self.variants)
        
        if self.assignments:
            confidences = [a.confidence for a in self.assignments]
            self.stats['average_confidence'] = np.mean(confidences)
            self.stats['min_confidence'] = np.min(confidences)
            self.stats['max_confidence'] = np.max(confidences)
        
        # Update group variant counts
        for group_id in self.product_groups:
            variant_count = sum(1 for v in self.variants.values() if v.group_id == group_id)
            self.product_groups[group_id].variant_count = variant_count
    
    def export_results(self, output_dir: str = 'output') -> None:
        """Export results to JSON and CSV files"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Export product groups
        groups_data = {
            'product_groups': [asdict(g) for g in self.product_groups.values()]
        }
        with open(f'{output_dir}/product_groups.json', 'w') as f:
            json.dump(groups_data, f, indent=2)
        
        # Export variants
        variants_data = {
            'variants': [asdict(v) for v in self.variants.values()]
        }
        with open(f'{output_dir}/variants.json', 'w') as f:
            json.dump(variants_data, f, indent=2)
        
        # Export assignments
        assignments_df = pd.DataFrame([asdict(a) for a in self.assignments])
        assignments_df['evidence'] = assignments_df['evidence'].apply(lambda x: ','.join(x))
        assignments_df.to_csv(f'{output_dir}/assignments.csv', index=False)
        
        # Export summary statistics
        with open(f'{output_dir}/summary.json', 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        logger.info(f"Results exported to {output_dir}/")

