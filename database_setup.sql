-- Table to store every individual IC50 calculation
CREATE TABLE ic50_history (
    id SERIAL PRIMARY KEY,
    drug_name VARCHAR(50) NOT NULL,
    ic50_value FLOAT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Table to store the ranked organoid results
CREATE TABLE organoid_results (
    id SERIAL PRIMARY KEY,
    organoid_id VARCHAR(100) UNIQUE NOT NULL,
    ic50_5fu FLOAT NOT NULL,
    z_score_5fu FLOAT NOT NULL,
    ic50_irinotecan FLOAT NOT NULL,
    z_score_irinotecan FLOAT NOT NULL,
    ic50_oxaliplatin FLOAT NOT NULL,
    z_score_oxaliplatin FLOAT NOT NULL,
    composite_score FLOAT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);