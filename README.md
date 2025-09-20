# Product Hierarchy Classifier
## Updates

---

# Product Hierarchy Processor

A Python tool to process product data and generate hierarchical product groups and variants with configurable granularity.

---

## Features

* Group products by **coarse, fine, or superfine** levels.
* Extract variant-level details including **color, RAM, storage, screen size, silicon**.
* Outputs include:

  * `group_id` → product family/group
  * `variant_id` → detailed configuration
  * `confidence` → matching confidence
  * `evidence` → features used for matching

---

## Granularity Levels

| Level         | Description                                                  | Example                 |
| ------------- | ------------------------------------------------------------ | ----------------------- |
| **Superfine** | Most detailed; includes full model + generation/year         | `asus_tuf_ryzen_7_2024` |
| **Fine**      | Moderate detail; removes year/generation, but keeps variants | `asus_tuf_ryzen_7`      |
| **Coarse**    | Broad grouping; only family-level                            | `asus_tuf`              |

**Variant IDs** always include configuration: color, RAM, storage, screen size, silicon.

---

## Example Output

**Superfine**

```
wmt_9969804547,asus_tuf_ryzen_7_2024,asus_tuf_ryzen_7_2024/config:black_8_512/size:15.6/silicon:amd_ryzen7_nvidia_generic,0.9999,"brand_match,model_match,config_extracted,size_extracted,silicon_extracted"
```

**Fine**

```
wmt_9969804547,asus_tuf_ryzen_7,asus_tuf_ryzen_7/config:black_8_512/size:15.6/silicon:amd_ryzen7_nvidia_generic,0.9999,"brand_match,model_match,config_extracted,size_extracted,silicon_extracted"
```

**Coarse**

```
wmt_9969804547,asus_tuf,asus_tuf/config:black_8_512/size:15.6/silicon:amd_ryzen7_nvidia_generic,0.9999,"brand_match,model_match,config_extracted,size_extracted,silicon_extracted"
```

---

## Installation

```bash
git clone <repo-url>
cd <repo-directory>
pip install -r requirements.txt
```

---

## Usage

```bash
python3 modified_code-submission.py --input ./merged_output.csv --output final_output_fine --coarse_level "fine"
```

**Arguments:**

* `--input` → Path to input CSV file
* `--output` → Directory to save output
* `--coarse_level` → Granularity level: `"coarse"`, `"fine"`, or `"superfine"`

---

## Notes

* **Confidence** is computed for each product assignment.
* **Evidence** lists which features were used for grouping.
* Recommended default for general use: `"fine"` level.

---






This assignment asks you to build a system that transforms unstructured product listings into a structured hierarchy.

## Quick Start

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   # Add any additional libraries you need
   ```

2. **Run the sample template (optional)**
   ```bash
   python sample_solution_template.py --input ../products-export-*.csv --output output --sample 100
   ```

3. **Explore the data**
   ```python
   import pandas as pd
   import json
   
   df = pd.read_csv('../products-export-*.csv')
   print(f"Dataset shape: {df.shape}")
   print("\nColumns:", df.columns.tolist())
   
   # Look at the JSON structure in details column
   sample_details = json.loads(df['details'].iloc[0])
   print("\nSample details structure:", sample_details.keys())
   ```

## Assignment Details

See [ASSIGNMENT.md](ASSIGNMENT.md) for the complete task description, requirements, and evaluation criteria.

## Submission Structure

Your solution should follow this structure (or create your own):

```
your-solution/
├── README.md              # Your approach and setup instructions
├── requirements.txt       # Your dependencies
├── src/                   # Your source code
└── output/               # Generated results
    ├── product_groups.json
    ├── variants.json
    ├── assignments.csv
    └── summary.json
```

## Template Usage

The `sample_solution_template.py` is completely optional. You can:
- Use it as a starting point
- Modify it extensively
- Ignore it completely and build your own solution
- Use a different programming language (Node.js, etc.)

We evaluate based on results and code quality, not conformity to any template.

## Tips

- Start by exploring the data structure in the `details` column
- Focus on getting a basic end-to-end pipeline working first
- Handle missing/malformed data gracefully
- Make your variant IDs deterministic (same input → same ID)
- Document your approach and key decisions

Good luck!
