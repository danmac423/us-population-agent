-- =====================================================================
-- 0. Setup
-- =====================================================================
USE WAREHOUSE US_POPULATION_WH;

CREATE DATABASE IF NOT EXISTS CENSUS_APP_DB;
CREATE SCHEMA IF NOT EXISTS CENSUS_APP_DB.PUBLIC;
USE SCHEMA CENSUS_APP_DB.PUBLIC;

-- =====================================================================
-- 1. Exploration
-- =====================================================================


SHOW TABLES IN SCHEMA US_OPEN_CENSUS_DATA_NEIGHBORHOOD_INSIGHTS_FREE_DATASET.PUBLIC;

SELECT * FROM US_OPEN_CENSUS_DATA_NEIGHBORHOOD_INSIGHTS_FREE_DATASET.PUBLIC."2020_METADATA_CBG_FIELD_DESCRIPTIONS" LIMIT 15;
SELECT * FROM US_OPEN_CENSUS_DATA_NEIGHBORHOOD_INSIGHTS_FREE_DATASET.PUBLIC."2020_METADATA_CBG_GEOGRAPHIC_DATA" LIMIT 5;
SELECT * FROM US_OPEN_CENSUS_DATA_NEIGHBORHOOD_INSIGHTS_FREE_DATASET.PUBLIC."2020_METADATA_CBG_FIPS_CODES" LIMIT 5;

-- =====================================================================
-- 2. Clean metadata table (one row per census estimate column)
-- =====================================================================
CREATE OR REPLACE TABLE CENSUS_APP_DB.PUBLIC.CLEAN_METADATA AS
SELECT
    TABLE_ID AS COLUMN_NAME,
    '2020_CBG_' || SUBSTR(TABLE_ID, 1, 3) AS DATA_TABLE_NAME,
    TABLE_TITLE,
    TABLE_TOPICS,
    -- Combine all 10 field levels for full context (NULL levels are skipped)
    'Column: ' || TABLE_ID ||
    ' | Table: 2020_CBG_' || SUBSTR(TABLE_ID, 1, 3) ||
    ' | Topic: ' || COALESCE(TABLE_TITLE, '') || ' (' || COALESCE(TABLE_TOPICS, '') || ')' ||
    ' | Universe: ' || COALESCE(TABLE_UNIVERSE, '') ||
    ' | Meaning: ' || ARRAY_TO_STRING(ARRAY_CONSTRUCT_COMPACT(
        FIELD_LEVEL_1, FIELD_LEVEL_2, FIELD_LEVEL_3, FIELD_LEVEL_4, FIELD_LEVEL_5,
        FIELD_LEVEL_6, FIELD_LEVEL_7, FIELD_LEVEL_8,
        "FIELD_LEVELl_9",  -- column name is misspelled in the source table
        FIELD_LEVEL_10
    ), ' > ') AS SEARCH_TEXT
FROM US_OPEN_CENSUS_DATA_NEIGHBORHOOD_INSIGHTS_FREE_DATASET.PUBLIC."2020_METADATA_CBG_FIELD_DESCRIPTIONS"
WHERE TABLE_ID LIKE '%e%';  -- estimates only

SELECT * FROM CENSUS_APP_DB.PUBLIC.CLEAN_METADATA LIMIT 100;

-- =====================================================================
-- 3. Cortex Search service (vector search over SEARCH_TEXT)
-- =====================================================================
CREATE OR REPLACE CORTEX SEARCH SERVICE CENSUS_APP_DB.PUBLIC.CENSUS_METADATA_SEARCH
  ON SEARCH_TEXT
  ATTRIBUTES DATA_TABLE_NAME, TABLE_TOPICS
  WAREHOUSE = US_POPULATION_WH
  TARGET_LAG = '1 day'
  EMBEDDING_MODEL = 'snowflake-arctic-embed-l-v2.0'
AS
  SELECT COLUMN_NAME, DATA_TABLE_NAME, TABLE_TITLE, TABLE_TOPICS, SEARCH_TEXT
  FROM CENSUS_APP_DB.PUBLIC.CLEAN_METADATA;

-- Test search
SELECT PARSE_JSON(
  SNOWFLAKE.CORTEX.SEARCH_PREVIEW(
    'CENSUS_APP_DB.PUBLIC.CENSUS_METADATA_SEARCH',
    '{"query": "number of renters paying more than 30% of income on rent",
      "columns": ["COLUMN_NAME", "DATA_TABLE_NAME", "SEARCH_TEXT"],
      "limit": 5}'
  )
):results AS results;

SELECT PARSE_JSON(
  SNOWFLAKE.CORTEX.SEARCH_PREVIEW(
    'CENSUS_APP_DB.PUBLIC.CENSUS_METADATA_SEARCH',
    '{"query": "poverty rate in New York?",
      "columns": ["COLUMN_NAME", "DATA_TABLE_NAME", "SEARCH_TEXT"],
      "limit": 100}'
  )
):results AS results;


-- =====================================================================
-- 4. Geography view (block group -> state, county, land/water area, lat/long)
-- =====================================================================
CREATE OR REPLACE VIEW CENSUS_APP_DB.PUBLIC.V_GEOGRAPHY AS
SELECT g.CENSUS_BLOCK_GROUP,
       f.STATE  AS STATE_ABBR,
       f.COUNTY AS COUNTY_NAME,
       g.AMOUNT_LAND, g.AMOUNT_WATER, g.LATITUDE, g.LONGITUDE
FROM US_OPEN_CENSUS_DATA_NEIGHBORHOOD_INSIGHTS_FREE_DATASET.PUBLIC."2020_METADATA_CBG_GEOGRAPHIC_DATA" g
LEFT JOIN US_OPEN_CENSUS_DATA_NEIGHBORHOOD_INSIGHTS_FREE_DATASET.PUBLIC."2020_METADATA_CBG_FIPS_CODES" f
  ON SUBSTR(g.CENSUS_BLOCK_GROUP, 1, 2) = f.STATE_FIPS
 AND SUBSTR(g.CENSUS_BLOCK_GROUP, 3, 3) = f.COUNTY_FIPS;


-- =====================================================================
-- 5. Search -> metadata lookup (top hit -> all columns of its table)
-- =====================================================================
WITH top_hit AS (
  SELECT PARSE_JSON(SNOWFLAKE.CORTEX.SEARCH_PREVIEW(
    'CENSUS_APP_DB.PUBLIC.CENSUS_METADATA_SEARCH',
    '{"query": "number of renters paying more than 30% of income on rent",
      "columns": ["COLUMN_NAME"], "limit": 1}'
  )):results[0]:COLUMN_NAME::STRING AS COLUMN_NAME
),
prefix AS (
  SELECT REGEXP_SUBSTR(COLUMN_NAME, '^[A-Z0-9]+e') AS P FROM top_hit   -- 'B25074e'
)
SELECT m.COLUMN_NAME, m.SEARCH_TEXT
FROM CENSUS_APP_DB.PUBLIC.CLEAN_METADATA m, prefix p
WHERE m.COLUMN_NAME LIKE p.P || '%'
ORDER BY TRY_TO_NUMBER(SUBSTR(m.COLUMN_NAME, LENGTH(p.P) + 1));