-- STEP 2 of 3. In pgAdmin: right-click the new "expense_db" database ->
-- Query Tool (so you are connected to expense_db, not postgres), then run
-- this whole file at once (F5).
--
-- The COMMENT lines are not decoration: the agent reads them and puts them in
-- the model's prompt, which tells the model the date format, the currency and
-- the allowed category values.

CREATE TABLE IF NOT EXISTS expenses (
    id         BIGSERIAL PRIMARY KEY,
    spent_on   DATE          NOT NULL DEFAULT CURRENT_DATE,
    amount     NUMERIC(12,2) NOT NULL CHECK (amount > 0),
    category   TEXT          NOT NULL DEFAULT 'other'
               CHECK (category IN ('food', 'transport', 'bills', 'shopping', 'health',
                                   'education', 'entertainment', 'transfer', 'other')),
    note       TEXT,
    created_at TIMESTAMPTZ   NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_expenses_spent_on ON expenses (spent_on);

COMMENT ON TABLE  expenses            IS 'One row per expense';
COMMENT ON COLUMN expenses.id         IS 'auto-generated, never set it';
COMMENT ON COLUMN expenses.spent_on   IS 'date of the expense, YYYY-MM-DD';
COMMENT ON COLUMN expenses.amount     IS 'amount in BDT, always positive';
COMMENT ON COLUMN expenses.category   IS 'one of: food, transport, bills, shopping, health, education, entertainment, transfer, other';
COMMENT ON COLUMN expenses.note       IS 'short free text such as lunch or rickshaw';
COMMENT ON COLUMN expenses.created_at IS 'row creation time, auto-generated, never set it';

-- The agent connects as expense_agent, which can only read and add rows.
-- It cannot update, delete or drop anything, even if the model tried to.
GRANT USAGE ON SCHEMA public TO expense_agent;
GRANT SELECT, INSERT ON expenses TO expense_agent;
GRANT USAGE ON SEQUENCE expenses_id_seq TO expense_agent;
