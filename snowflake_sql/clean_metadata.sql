-- Column-level corpus: one row per ACS estimate column, with a readable description
CREATE OR REPLACE TABLE CENSUS_APP_DB.PUBLIC.CLEAN_METADATA AS
SELECT
    TABLE_NUMBER AS TABLE_ID,       -- e.g. B25074CENSUS_APP_DB.PUBLIC.V_GEOGRAPHY (joins to TABLE_CATALOG)
    TABLE_ID AS COLUMN_NAME,        -- e.g. B25074e60
    '2020_CBG_' || SUBSTR(TABLE_ID, 1, 3) AS DATA_TABLE_NAME,
    
    -- Text passed to the LLM
    'Column: ' || TABLE_ID || 
    ' | Table: 2020_CBG_' || SUBSTR(TABLE_ID, 1, 3) ||
    ' | Topic: ' || COALESCE(TABLE_TOPICS, '') || 
    ' | Title: ' || COALESCE(TABLE_TITLE, '') || 
    ' | Universe: ' || COALESCE(TABLE_UNIVERSE, '') ||
    ' | Meaning: ' ||
    -- ARRAY_CONSTRUCT_COMPACT skips NULL values
    ARRAY_TO_STRING(ARRAY_CONSTRUCT_COMPACT(
        NULLIF(FIELD_LEVEL_2, ''),
        NULLIF(FIELD_LEVEL_3, ''),
        NULLIF(FIELD_LEVEL_4, ''),
        NULLIF(FIELD_LEVEL_5, ''),
        NULLIF(FIELD_LEVEL_6, ''),
        NULLIF(FIELD_LEVEL_7, ''),
        NULLIF(FIELD_LEVEL_8, ''),
        NULLIF("FIELD_LEVELl_9", ''), -- Column name is misspelled in source
        NULLIF(FIELD_LEVEL_10, '')
    ), ' > ') AS COLUMN_DESCRIPTION
FROM US_OPEN_CENSUS_DATA_NEIGHBORHOOD_INSIGHTS_FREE_DATASET.PUBLIC."2020_METADATA_CBG_FIELD_DESCRIPTIONS"
WHERE TABLE_ID LIKE '%e%'             -- Take only the estimates (the actual data, discarding the 'm' margins of errors)
  AND TABLE_NUMBER NOT LIKE 'B99%';       -- Exclude table B99 (Imputed/Allocated answers - data quality)