"""Single description of the data the assistant can answer from. Shared by the classifier and the reply prompts."""

DATASET_SUMMARY = """- Source: American Community Survey (ACS) 2020 5-year estimates, US Census.
- Unit: Census Block Group (CBG). Values can be aggregated to:
  * county (COUNTY_NAME, e.g. "Autauga County", "Orleans Parish")
  * state (STATE_ABBR, includes DC and Puerto Rico)
  * whole country
- Geography also has land/water area and CBG centroid latitude/longitude.
- Topics (ACS tables): population, age, sex, race, ethnicity, ancestry, citizenship,
  nativity, language, education, school enrollment, employment, occupation, industry,
  commuting, income, earnings, poverty, public assistance, health insurance, disability,
  veterans, households, families, marital status, housing units, occupancy, tenure
  (own/rent), rent, home value, mortgage, housing costs, vehicles, internet/computer access.
- Supported operations: totals, counts, percentages/shares, comparisons between
  areas, rankings (e.g. top 10 counties), values per CBG.
- NOT available: other years or trends over time, data outside the US, cities/ZIP codes/
  metro areas/congressional districts as filters, individual-level records, forecasts,
  margins-of-error analysis, non-census data (crime, prices today, weather, elections results)."""
