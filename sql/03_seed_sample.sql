-- STEP 3 of 3 (optional). Run in the Query Tool of expense_db.
-- Sample data with dates relative to today, so "last 10 days" questions
-- have something to find no matter when you run this.

INSERT INTO expenses (spent_on, amount, category, note) VALUES
    (CURRENT_DATE,      180.00, 'food',          'lunch'),
    (CURRENT_DATE,       60.00, 'transport',     'rickshaw'),
    (CURRENT_DATE - 1,  250.00, 'food',          'dinner with friends'),
    (CURRENT_DATE - 2,  500.00, 'transfer',      'bkash to Rafi'),
    (CURRENT_DATE - 3,  120.00, 'food',          'tea and snacks'),
    (CURRENT_DATE - 4, 1200.00, 'bills',         'internet'),
    (CURRENT_DATE - 6,  350.00, 'shopping',      'notebook and pens'),
    (CURRENT_DATE - 8,   90.00, 'transport',     'bus'),
    (CURRENT_DATE - 9,  300.00, 'entertainment', 'movie'),
    (CURRENT_DATE - 12, 220.00, 'food',          'lunch'),
    (CURRENT_DATE - 15, 800.00, 'health',        'pharmacy'),
    (CURRENT_DATE - 20, 450.00, 'education',     'course material');
