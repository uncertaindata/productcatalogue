# Product Hierarchy Classifier Assignment
## Building a Modern Product Knowledge Graph

### 🎯 Assignment Overview

**Duration:** 10-12 hours  
**Focus:** ML Engineering, Data Processing, System Design

You will build a system that transforms unstructured e-commerce product listings into a structured hierarchy, creating the foundation for a Product Knowledge Graph. This is a real-world problem that major e-commerce platforms face when organizing millions of products from different sellers.

### 📋 Business Context

Our e-commerce platform aggregates products from multiple sources (Amazon, Walmart, etc.). The same product appears multiple times with different titles, descriptions, and formats. We need to:

1. **Group products into families** (ProductGroups) - e.g., all "MacBook Air M2" laptops
2. **Identify specific configurations** (Variants) - e.g., "256GB/Silver" vs "512GB/Space Gray"  
3. **Extract normalized attributes** (Variant Axes) - the key dimensions that differentiate products

This structured hierarchy enables:
- Better search and filtering
- Accurate price comparisons
- Inventory management
- Personalized recommendations

### 🏗️ The Task

Transform raw product data into a three-tier hierarchy:

```
ProductGroup (Family)
    ↓
Variant (Specific Configuration)
    ↓
SKU (Individual Listing)
```

#### Example Transformation:

**Input:** Three messy product listings
```
1. "Apple MacBook Air 13.6 inch M2 Chip 256GB SSD Silver 2024"
2. "MacBook Air M2 13-inch 256 GB Silver - Latest Model"
3. "Apple MacBook Air 13.6" M2 256GB Space Gray"
```

**Output:** Structured hierarchy
```
ProductGroup: "apple_macbook_air_m2_2024"
├── Variant: ".../config:8_256_silver" (listings 1 & 2)
└── Variant: ".../config:8_256_space_gray" (listing 3)
```

### 📊 The Dataset

You'll work with `products-export-*.csv` containing:
- **~3,000 products** (laptops and TVs primarily)
- **Raw JSON data** in the `details` column from web scraping
- **Multiple sellers** with inconsistent naming conventions
- **Some accessories** that should be identified and handled appropriately

Key columns:
- `product_id`: Unique identifier
- `seller_id`: Source marketplace
- `name`: Product title
- `details`: JSON blob with specifications, pricing, variations
- `category`, `sub_category`: High-level classification

### 🎯 Core Requirements

#### 1. Data Processing Pipeline (2-3 hours)

Build a robust pipeline to:
- Parse complex nested JSON from the `details` column
- Extract and normalize product specifications
- Handle missing, malformed, or inconsistent data
- Clean and standardize text fields

**Deliverable:** Clean, normalized dataset with extracted features

#### 2. Variant Axis Extraction (2-3 hours)

Implement logic to identify and extract the six standard variant axes:

| Axis | Purpose | Example Extraction |
|------|---------|-------------------|
| **config** | Core configuration | RAM: 8GB, Storage: 256GB, Color: Silver |
| **size** | Physical dimensions | Screen: 13.6", TV: 65" |
| **silicon** | Processing components | CPU: M2, GPU: Intel Iris |
| **region** | Market-specific | Voltage: 120V, Plug: Type-A |
| **carrier** | Network compatibility | Unlocked, Verizon (if applicable) |
| **packaging** | Item state | Condition: New, Bundle: No |

**Deliverable:** Normalized axis values for each product

#### 3. ProductGroup Creation (2-3 hours)

Develop algorithm to:
- Cluster similar products into families
- Identify shared characteristics (base_specs)
- Generate stable, deterministic group IDs
- Handle edge cases (accessories, bundles)

**Deliverable:** List of ProductGroups with their base specifications

#### 4. Variant Assignment (2-3 hours)

Create system to:
- Generate deterministic variant IDs from axes
- Assign products to correct variants
- Calculate confidence scores for assignments
- Handle ambiguous cases

**Deliverable:** Product-to-variant mapping with confidence scores

### 📤 Expected Output Format

#### 1. ProductGroups JSON
```json
{
  "product_groups": [
    {
      "group_id": "apple_macbook_air_m2_2024",
      "brand": "apple",
      "family": "MacBook Air M2",
      "generation": "2024",
      "base_specs": {
        "processor_family": "Apple M2",
        "screen_size_inches": 13.6,
        "form_factor": "laptop"
      },
      "variant_count": 12,
      "product_count": 45
    }
  ]
}
```

#### 2. Variants JSON
```json
{
  "variants": [
    {
      "variant_id": "apple_macbook_air_m2_2024/config:8_256_silver/size:13.6",
      "group_id": "apple_macbook_air_m2_2024",
      "axes": {
        "config": {
          "ram_gb": 8,
          "storage_gb": 256,
          "color": "silver"
        },
        "size": {
          "screen_inches": 13.6
        }
      },
      "product_count": 5
    }
  ]
}
```

#### 3. Assignments CSV
```csv
product_id,group_id,variant_id,confidence,evidence
ABC123,apple_macbook_air_m2_2024,apple_macbook_air_m2_2024/config:8_256_silver,0.95,"exact_model_match,spec_match"
```

#### 4. Summary Statistics
```json
{
  "total_products": 3000,
  "total_groups": 150,
  "total_variants": 450,
  "products_assigned": 2850,
  "products_unassigned": 150,
  "average_confidence": 0.87,
  "processing_time_seconds": 45.2
}
```

### 🔧 Technical Guidelines

#### Required Functionality

Your solution MUST:
1. Process all products without crashing
2. Handle malformed/missing data gracefully
3. Produce valid JSON/CSV outputs
4. Include confidence scoring
5. Be reproducible (same input → same output)

#### Recommended Approach

1. **Start simple**: Get a basic pipeline working end-to-end
2. **Iterate**: Improve accuracy and handle edge cases
3. **Validate**: Check your outputs make business sense
4. **Document**: Explain your design decisions

#### Technology Choices

**Required:**
- Python 3.8+ or Node.js 16+
- Standard data processing libraries (pandas, numpy, etc.)

**Optional:**
- Scikit-learn for clustering/classification
- Sentence-transformers for embeddings (optional)
- Fuzzy matching libraries (fuzzywuzzy, rapidfuzz)

### 📊 Evaluation Criteria

#### Core Competencies (70%)

1. **Data Processing (25%)**
   - Correctly parses complex JSON structures
   - Handles data quality issues gracefully
   - Normalizes values appropriately

2. **Axis Extraction (25%)**
   - Accurately identifies variant dimensions
   - Normalizes values (1TB → 1024GB)
   - Handles missing/ambiguous data

3. **Hierarchy Building (20%)**
   - Logical ProductGroup creation
   - Correct variant differentiation
   - Deterministic ID generation

#### Code Quality (20%)

- Clean, modular architecture
- Appropriate error handling
- Clear documentation
- Efficient algorithms

#### Analysis & Insights (10%)

- Summary statistics
- Confidence scoring methodology
- Identified data quality issues
- Recommendations for improvement

### 🌟 Bonus Challenges (Optional)

Worth extra credit but not required:

1. **ML-Powered Matching**
   - Use embeddings for similarity scoring
   - Implement learned classification model
   - Show performance improvements

2. **Bundle Detection**
   - Identify product bundles
   - Separate accessories from main products
   - Handle multi-packs correctly

3. **Advanced Analysis**
   - Data quality scoring per seller
   - Identify problematic patterns
   - Suggest data collection improvements

### 📁 Submission Requirements

Your submission should include:

```
product-hierarchy-classifier/
├── README.md                 # Setup instructions & approach explanation
├── requirements.txt          # Python dependencies (or package.json)
├── src/
│   ├── pipeline.py          # Main processing pipeline
│   ├── extractors.py        # Axis extraction logic
│   ├── classifiers.py       # Group/variant assignment
│   └── utils.py             # Helper functions
├── output/
│   ├── product_groups.json  # ProductGroup definitions
│   ├── variants.json        # Variant definitions
│   ├── assignments.csv      # Product-to-variant mapping
│   └── summary.json         # Statistics and metrics
└── notebooks/               # Optional: exploratory analysis
    └── analysis.ipynb
```

### 🚀 Getting Started

1. **Load and explore the data**
   ```python
   import pandas as pd
   import json
   
   df = pd.read_csv('products-export-*.csv')
   # Parse the 'details' column JSON
   df['details_parsed'] = df['details'].apply(json.loads)
   ```

2. **Extract key attributes**
   ```python
   def extract_storage(details):
       # Look in specifications, title, features
       # Normalize TB to GB, handle different formats
       pass
   ```

3. **Build variant axes**
   ```python
   def build_variant_axes(product):
       return {
           'config': extract_config(product),
           'size': extract_size(product),
           'silicon': extract_silicon(product)
       }
   ```

4. **Create hierarchy**
   ```python
   def assign_to_group(product, existing_groups):
       # Use similarity scoring to find best match
       # Or create new group if no match found
       pass
   ```

### ⚠️ Important Notes

1. **Accessories Handling**: Some products are accessories (cases, cables). These should be identified but can be excluded from the main hierarchy or placed in separate groups.

2. **Data Quality**: The raw data has inconsistencies. Part of the challenge is building a robust system that handles real-world messiness.

3. **Performance**: While not the primary focus, your solution should process the full dataset in reasonable time (<5 minutes).

4. **Deterministic IDs**: Variant IDs should be generated from normalized axes so the same configuration always gets the same ID.

### ❓ FAQ

**Q: Should I handle products other than laptops and TVs?**  
A: Focus primarily on laptops and TVs. Identify accessories but handling them is optional.

**Q: How should I handle products with missing specifications?**  
A: Assign them with lower confidence scores and flag for review.

**Q: Can I use external APIs or pretrained models?**  
A: Yes, but document what you're using and why. The core logic should be your own.

**Q: Should color variations be separate variants?**  
A: Yes, color is part of the `config` axis and creates different variants.

**Q: How deterministic should the IDs be?**  
A: Very. Same input data should always generate the same IDs.

### 📧 Questions?

If you need clarification on requirements, please reach out. We're looking for practical problem-solving skills and clean implementation rather than perfect accuracy.

Good luck! We're excited to see your approach to building this Product Knowledge Graph foundation.

---