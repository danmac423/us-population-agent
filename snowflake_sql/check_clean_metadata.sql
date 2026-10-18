-- Show everything that CLEAN_METADATA contains
SELECT * FROM CENSUS_APP_DB.PUBLIC.CLEAN_METADATA;


SELECT * FROM CENSUS_APP_DB.PUBLIC.CLEAN_METADATA WHERE TABLE_ID IN ('B29003', 'B01001');

-- ===== DATA VALIDATION =====

-- 1) Basic integrity: counts, duplicates, NULLs
SELECT COUNT(*)                                        AS n_rows,           -- expect 3782
       COUNT(DISTINCT TABLE_ID)                        AS n_tables,         -- expect 298
       COUNT(*) - COUNT(DISTINCT COLUMN_NAME)          AS dup_columns,      -- expect 0
       COUNT_IF(TABLE_ID IS NULL
             OR COLUMN_NAME IS NULL
             OR COLUMN_DESCRIPTION IS NULL)            AS null_values       -- expect 0
FROM CENSUS_APP_DB.PUBLIC.CLEAN_METADATA;

-- 2) Filters worked: no B99 tables, no margin-of-error ('m') columns
SELECT COUNT_IF(TABLE_ID LIKE 'B99%')                  AS b99_rows,         -- expect 0
       COUNT_IF(COLUMN_NAME NOT LIKE '%e%')            AS non_estimates     -- expect 0
FROM CENSUS_APP_DB.PUBLIC.CLEAN_METADATA;

-- 3) Derived columns are consistent
SELECT COUNT_IF(NOT STARTSWITH(COLUMN_NAME, TABLE_ID || 'e'))              AS id_mismatch,     -- expect 0
       COUNT_IF(DATA_TABLE_NAME <> '2020_CBG_' || SUBSTR(TABLE_ID, 1, 3)) AS bad_table_name,  -- expect 0
       COUNT_IF(COLUMN_DESCRIPTION LIKE '%Meaning: ')                     AS empty_meaning    -- expect 0
FROM CENSUS_APP_DB.PUBLIC.CLEAN_METADATA;

-- 4) Matches TABLE_CATALOG: same column total, same set of tables
SELECT (SELECT COUNT(*)       FROM CENSUS_APP_DB.PUBLIC.CLEAN_METADATA) AS metadata_cols,  -- expect 3782
       (SELECT SUM(N_COLUMNS) FROM CENSUS_APP_DB.PUBLIC.TABLE_CATALOG)  AS catalog_cols,   -- expect 3782
       (SELECT COUNT(*) FROM CENSUS_APP_DB.PUBLIC.TABLE_CATALOG c
        WHERE c.TABLE_ID NOT IN (SELECT TABLE_ID FROM CENSUS_APP_DB.PUBLIC.CLEAN_METADATA)) AS missing_tables;  -- expect 0

-- 5) Eyeball one table: descriptions should read naturally, in census order
SELECT COLUMN_NAME, COLUMN_DESCRIPTION
FROM CENSUS_APP_DB.PUBLIC.CLEAN_METADATA
WHERE TABLE_ID = 'B25074'
ORDER BY TRY_TO_NUMBER(REGEXP_SUBSTR(COLUMN_NAME, 'e([0-9]+)$', 1, 1, 'e', 1));