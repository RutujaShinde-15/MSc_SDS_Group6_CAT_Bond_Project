# CFDID: China Flood Disaster Impact Database, 1994–2023

CFDID contains 1,716 flood-impact records for the Chinese mainland during 1994–2023: 1,391 L1, 300 L2 and 25 L3 records, with 22 fields. Five categories are included: rainstorm flood, typhoon, snowmelt flood, ice-jam flood and dam-break flood.

Preprint: [Cui et al. (2026), doi:10.5194/essd-2026-607](https://doi.org/10.5194/essd-2026-607). [Preprint and discussion page](https://essd.copernicus.org/preprints/essd-2026-607/).

**Files**

| File or directory | Contents |
|---|---|
| `China Flood Disaster Impact Database.csv` | Database in UTF-8 CSV format; one record per row. |
| `China Flood Disaster Impact Database.xlsx` | Excel database, worksheet `All events`. |
| `data_dictionary.csv` | Definitions, types, units and requirements for the 22 database fields. |
| `requirements.txt` | Python dependencies for the workflow and validator. |
| `Official national annual totals_1994-2023.xlsx` | Annual national benchmarks for the five core impact indicators. |
| `Cross records of CFDID and EM-DAT.xlsx` | Bidirectional lists of matched and unmatched record IDs. |
| `main code/6_Validate.py` | Validator for the released XLSX database. |
| `main code/` | Workflow scripts 1–5 and the database validator, script 6. |
| `main code/example/` | The 2022 source extract and saved intermediate processing outputs. |

**Reading and aggregating the database**

- L1 denotes a parent event; L2 and L3 denote nested sub-events. Text IDs follow `YYYY-NNN`, `YYYY-NNN-SS` and `YYYY-NNN-SS-GG`; removing the last suffix gives the parent ID. Use **L1 only** for national counts and sums to avoid double counting.
- Empty CSV fields and blank Excel cells mean information was not reported, **not zero**. Keep unknown date components blank. Read CSV using `utf-8-sig`.
- `Province` lists use commas; `Data source` lists use commas or `、`. Trim spaces after splitting. Multi-province impacts cannot be assigned in full to every listed province.
- Units: persons for deaths/missing; thousand persons for affected population; thousand hectares for cropland; thousand rooms for dwellings; million CNY for losses. Do not add overlapping cropland indicators or the two price representations of losses.
- At least one of five core indicators must be positive: deaths/missing, affected population, affected cropland, collapsed dwellings or current-price losses. Other impact fields may be missing; individual reported zeros are valid.
- `Loss_2023 = current-price loss × CPI_2023 / CPI_event_year`, with CPI indices on a common base from the China Statistical Yearbooks. Use 2023-price columns for comparisons.

**Sources, benchmarks and limitations**

The 66 sources comprise the China Water Resources Yearbook (25; event years 1994–2018), Yearbook of Meteorological Disasters in China (22; 2002–2023), and annual Top Ten Natural Disasters in China reports (19; 2005–2023). Conflict-resolution priority was Top Ten reports, meteorological yearbooks, then water resources yearbooks.

National benchmarks use water resources yearbooks for 1994–2001 and meteorological yearbooks for 2002–2023. Percentages equal summed reported L1 impacts divided by the corresponding official totals × 100; the 30-year result uses sums, not averaged annual percentages.

Before fusion, manual checking evaluated 18 fields across 1,912 records from 1994–2023: 34,416 field instances and 99.25% accuracy, including correct missing values. This measures extraction agreement with sources, not completeness or source accuracy.

**CFDID and EM-DAT correspondences**

`CFDID_to_EMDAT` contains 569 matched and 822 unmatched CFDID L1 records; `EMDAT_to_CFDID` contains 313 matched and 120 unmatched EM-DAT records. `/` means no identified counterpart; multiple counterpart IDs are separated by `、` or commas. Split on either separator and trim spaces. The 653 ID pairs include one-to-many/many-to-many relationships. Count each original record once; do not sum impacts over expanded pairs.

Matching used dates, provinces and categories, then manual checking; typhoon names/numbers aided identification. Record `2018-0369-CHN` lacked sufficient location information and was excluded: Table 4 uses 434 EM-DAT records, whereas matching and Fig. 7d–f use 433. Unmatched does not establish absence from the other database.

The numerical comparison uses **deaths/missing ≥10 OR affected population ≥100 persons** (`>= 0.1` thousand persons). Missing values do not establish failure to meet a threshold. Emergency-declaration and international-assistance criteria are not evaluated.

External sources: [EM-DAT](https://www.emdat.be/) (downloaded 28 April 2026), [Wikimpacts 1.0](https://bolin.su.se/data/li-2025-wikimpacts-1.0.final) and [DFO v0.9.0](https://doi.org/10.5281/zenodo.19288171). EM-DAT selection covers China, 1994–2023: Flood, Landslide (wet), Rockfall (wet), Mudslide, Tropical cyclone and Glacial lake outburst flood. Obtain external impact tables from their providers; only the ID crosswalk is included here.

**Software and validation**

The reported study environment used Python 3.13.5. Python dependencies are listed in `requirements.txt`. Run the following commands from the directory containing this README:

```sh
python -m pip install -r requirements.txt
python "main code/6_Validate.py" "China Flood Disaster Impact Database.xlsx" --report validation_report.csv
```

Alternatively, run `6_Validate.py` directly in VS Code without arguments. It finds the database in the parent of `main code/` and saves `China Flood Disaster Impact Database_validation_report.csv` beside it, regardless of the terminal's working directory. The terminal prints the report's full path.

The validator checks XLSX structure, IDs, hierarchy, dates, categories and impacts without modifying the input; it writes errors/warnings to CSV. CPI consistency checks allow three-decimal rounding in both loss columns and require a common annual conversion factor, with factor 1 for 2023. They do not verify historical CPI values, source facts or CSV files. The current XLSX passed validation on 23 September 2026 with zero errors and zero warnings.

**Reusing the 2022 example**

Run workflow scripts **1–5** in order from the directory containing this README, using `python "main code/<script filename>"`. Their example paths are resolved relative to the scripts. Run **`6_Validate.py` separately on the released database**, using the command above. Saved intermediate files in `main code/example/` can be inspected without API calls.

| Script | Processing and handoff |
|---|---|
| `1_Document Preprocessing (OCR).py` | Calls MinerU; set `MINERU_API_TOKEN`. Results are saved under `main code/mineru_ocr_outputs/`. Review the OCR content and prepare `main code/example/2_OCR_output 2022.docx` for step 2. |
| `2_Rule-based Segmentation.py` | Converts the OCR Word document to text segments in Excel. |
| `3_Data Extraction.py` | Calls `deepseek-v4-pro`; set `DEEPSEEK_API_KEY`. Review extracted fields against the source. Input filenames must contain the event year. |
| `4_Data Fusion.py` | Uses the same LLM to assess candidate record relationships. Manually resolve conflicts and assign nesting; follow the structure of `main code/example/5_Manual-checked 2022.xlsx`, with each parent followed by its children. |
| `5_Data Post-processing.py` | Sorts records and assigns hierarchical IDs. |

**Version history, citation and contact**

- Initial release (2026): 1,716 records covering 1994–2023.
- September 2026 revision: corrected impact values and event dates; added national benchmarks, the EM-DAT crosswalk, validation and repository documentation. Record counts remain unchanged.
- Future annual releases will document added records, corrections and changes to fields or code. Consult the accompanying Zenodo record for the release version, date, dataset DOI and license; cite the version actually used.

Paper citation: Cui, S., Li, N., Hu, S., Wan, Z., Wang, J., Tang, J., Zhang, R., Li, Y., Yu, X., Tan, Y., Shan, W., and Zhao, J.: CFDID: a China Flood Disaster Impact Database from 1994 to 2023, Earth System Science Data Discussions [preprint], [https://doi.org/10.5194/essd-2026-607](https://doi.org/10.5194/essd-2026-607), 2026.

Contact: 
Shibo Cui, Department of Hydraulic Engineering, Tsinghua University, Beijing, China; [cuishibo@tsinghua.edu.cn].
Jianshi Zhao, Department of Hydraulic Engineering, Tsinghua University, Beijing, China; [zhaojianshi@tsinghua.edu.cn].
