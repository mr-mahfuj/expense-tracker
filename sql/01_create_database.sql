-- STEP 1 of 3. In pgAdmin: connect to the default "postgres" database as the
-- postgres superuser, open Query Tool.
--
-- Run each statement ONE AT A TIME: select it, then press F5.
-- (CREATE DATABASE cannot run inside a multi-statement batch.)
--
-- Before running: replace CHANGE_ME with a real password, and put the same
-- password in your .env file (DATABASE_URL).

CREATE ROLE expense_agent LOGIN PASSWORD 'CHANGE_ME';

CREATE DATABASE expense_db;

GRANT CONNECT ON DATABASE expense_db TO expense_agent;
