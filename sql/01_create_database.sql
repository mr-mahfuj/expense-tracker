-- STEP 1 of 3. In pgAdmin: connect to the default "postgres" database as the
-- postgres superuser, open Query Tool.
--
-- Run the CREATE DATABASE statement by itself (F5).
-- CREATE DATABASE cannot run inside a transaction block.
--
-- Uses the existing postgres login and its existing password from DATABASE_URL.
-- If expense-tracker-db already exists, skip this step and run 02_schema.sql
-- while connected to that database as postgres.

CREATE DATABASE "expense-tracker-db" OWNER postgres;
