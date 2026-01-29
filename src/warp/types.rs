/// SQL operators for filtering
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(dead_code)]
pub enum SqlOperator {
    Equals,
    NotEquals,
    GreaterThan,
    LessThan,
    GreaterThanOrEquals,
    LessThanOrEquals,
    Like,
    NotLike,
    Exists,
}

/// Logical operators for combining conditions
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(dead_code)]
pub enum SqlLogic {
    And,
    Or,
}

/// Type of SQL statement
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(dead_code)]
pub enum SqlStatementType {
    Condition,
    Group,
    Empty,
}

/// A single condition in a SQL filter
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum SqlValue {
    String(String),
    Number(f64),
}

#[derive(Debug, Clone)]
pub struct SqlCondition {
    pub key: String,
    pub operator: SqlOperator,
    pub value: Option<SqlValue>,
}

/// A SQL filter statement that can be a condition or a group of statements
#[derive(Debug, Clone)]
pub struct SqlStatement {
    pub statement_type: SqlStatementType,
    pub condition: Option<SqlCondition>,
    pub logic: Option<SqlLogic>,
    pub statements: Option<Vec<SqlStatement>>,
}
