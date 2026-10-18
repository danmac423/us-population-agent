-- Show everything that TABLE_CATALOG contains
SELECT * FROM CENSUS_APP_DB.PUBLIC.TABLE_CATALOG 
ORDER BY DATA_TABLE_NAME, TABLE_ID ASC;


-- ===== DATA VALIDATION =====

-- 1) Check number of rows in TABLE_CATALOG
SELECT COUNT(*) FROM CENSUS_APP_DB.PUBLIC.TABLE_CATALOG; 

-- 2) Basic integrity: counts, duplicates, NULLs
SELECT COUNT(*)                                   AS n_rows,    -- expect 298
       COUNT(*) - COUNT(DISTINCT TABLE_ID)        AS dup_ids,
       COUNT_IF(TABLE_ID IS NULL)                 AS null_ids,
       COUNT_IF(SEARCH_TEXT IS NULL)              AS null_search_text,
       COUNT_IF(BREAKDOWNS IS NULL)               AS no_breakdowns   -- expect only Total-only tables
FROM CENSUS_APP_DB.PUBLIC.TABLE_CATALOG;

-- 3) Tables without breakdowns should all be single-column
SELECT *
FROM CENSUS_APP_DB.PUBLIC.TABLE_CATALOG
WHERE BREAKDOWNS IS NULL;



-- ===== COMPARTE WITH SOURCE DATA =====

-- 1) Column totals must match the source (same filters as the build)
SELECT (SELECT SUM(N_COLUMNS) FROM CENSUS_APP_DB.PUBLIC.TABLE_CATALOG) AS catalog_cols,
       (SELECT COUNT(*)
        FROM US_OPEN_CENSUS_DATA_NEIGHBORHOOD_INSIGHTS_FREE_DATASET.PUBLIC."2020_METADATA_CBG_FIELD_DESCRIPTIONS"
        WHERE TABLE_ID LIKE '%e%' AND TABLE_NUMBER NOT LIKE 'B99%')    AS source_cols;


-- 2) Show orginal field descriptions data to compare with SEARCH_TEXT field in TABLE_CATALOG
SELECT * FROM US_OPEN_CENSUS_DATA_NEIGHBORHOOD_INSIGHTS_FREE_DATASET.PUBLIC."2020_METADATA_CBG_FIELD_DESCRIPTIONS" ORDER BY TABLE_ID ASC;


-- 3) Compare one table side by side
SELECT TABLE_ID, SEARCH_TEXT FROM CENSUS_APP_DB.PUBLIC.TABLE_CATALOG WHERE TABLE_ID = 'B25074';

SELECT * FROM US_OPEN_CENSUS_DATA_NEIGHBORHOOD_INSIGHTS_FREE_DATASET.PUBLIC."2020_METADATA_CBG_FIELD_DESCRIPTIONS"
WHERE TABLE_NUMBER = 'B25074' ORDER BY TABLE_ID;



-- ===== CORTEX SEARCH ======

-- Show results of CORTEX Search for given query
SELECT PARSE_JSON(
  SNOWFLAKE.CORTEX.SEARCH_PREVIEW(
    'CENSUS_APP_DB.PUBLIC.CENSUS_TABLE_SEARCH',
    '{"query": "poverty in Texas and California",
      "columns": ["TABLE_ID", "DATA_TABLE_NAME", "TABLE_TITLE", "SEARCH_TEXT"],
      "limit": 5}'
  )
):results AS results;

-- Search service is up and has indexed the latest data
SHOW CORTEX SEARCH SERVICES LIKE 'CENSUS_TABLE_SEARCH' IN SCHEMA CENSUS_APP_DB.PUBLIC;