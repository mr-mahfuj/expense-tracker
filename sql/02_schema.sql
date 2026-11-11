-- STEP 2 of 3. In pgAdmin: right-click the new "expense-tracker-db" database ->
-- Query Tool (so you are connected to expense-tracker-db, not postgres), then run
-- this whole file at once (F5).
-- Connect using the existing postgres login from DATABASE_URL.
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

-- The application connects as postgres, which already has full access.
-- No separate application role or grants are required. With this connection,
-- database permissions do not restrict the application to SELECT and INSERT.
