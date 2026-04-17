//! Filter builder that generates LanceDB `only_if()` SQL-like filter strings.
//!
//! Replaces the old `sql_generator.rs` + `SqlStatementInternal` system.
//! Filters operate on the flat metadata columns defined in the [`MetadataSchema`].

use serde::{Deserialize, Serialize};

/// A dynamically-typed metadata value.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MetadataValue {
    String(String),
    Number(f64),
    Bool(bool),
}

/// Comparison operators for filter conditions.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "napi", napi_derive::napi(string_enum))]
pub enum FilterOp {
    Eq,
    Ne,
    Gt,
    Lt,
    Gte,
    Lte,
    Like,
    NotLike,
    IsNotNull,
}

/// Logical combinators for groups of filters.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[cfg_attr(feature = "napi", napi_derive::napi(string_enum))]
pub enum FilterLogic {
    And,
    Or,
}

/// A single filter condition on a metadata column.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterCondition {
    pub field: String,
    pub op: FilterOp,
    pub value: Option<MetadataValue>,
}

/// A composable filter tree for metadata queries.
///
/// Can be a single condition, a group of conditions combined with AND/OR,
/// or empty (no filter).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Filter {
    Condition(FilterCondition),
    Group {
        logic: FilterLogic,
        filters: Vec<Filter>,
    },
    Empty,
}

impl Filter {
    /// Create an equality filter: `field = value`.
    pub fn eq(field: &str, value: MetadataValue) -> Self {
        Filter::Condition(FilterCondition {
            field: field.to_string(),
            op: FilterOp::Eq,
            value: Some(value),
        })
    }

    /// Create a NOT NULL filter: `field IS NOT NULL`.
    pub fn is_not_null(field: &str) -> Self {
        Filter::Condition(FilterCondition {
            field: field.to_string(),
            op: FilterOp::IsNotNull,
            value: None,
        })
    }

    /// Combine two filters with AND.
    pub fn and(self, other: Filter) -> Self {
        match self {
            Filter::Empty => other,
            Filter::Group {
                logic: FilterLogic::And,
                mut filters,
            } => {
                filters.push(other);
                Filter::Group {
                    logic: FilterLogic::And,
                    filters,
                }
            }
            _ => Filter::Group {
                logic: FilterLogic::And,
                filters: vec![self, other],
            },
        }
    }

    /// Combine two filters with OR.
    pub fn or(self, other: Filter) -> Self {
        match self {
            Filter::Empty => other,
            Filter::Group {
                logic: FilterLogic::Or,
                mut filters,
            } => {
                filters.push(other);
                Filter::Group {
                    logic: FilterLogic::Or,
                    filters,
                }
            }
            _ => Filter::Group {
                logic: FilterLogic::Or,
                filters: vec![self, other],
            },
        }
    }

    /// Convert this filter tree to a LanceDB `only_if()` SQL-like filter string.
    ///
    /// Returns an empty string for `Filter::Empty`.
    pub fn to_lance_filter(&self) -> String {
        match self {
            Filter::Empty => String::new(),
            Filter::Condition(c) => {
                let op_str = match c.op {
                    FilterOp::Eq => "=",
                    FilterOp::Ne => "!=",
                    FilterOp::Gt => ">",
                    FilterOp::Lt => "<",
                    FilterOp::Gte => ">=",
                    FilterOp::Lte => "<=",
                    FilterOp::Like => "LIKE",
                    FilterOp::NotLike => "NOT LIKE",
                    FilterOp::IsNotNull => "IS NOT NULL",
                };

                match (&c.value, &c.op) {
                    (_, FilterOp::IsNotNull) => {
                        format!("{} IS NOT NULL", c.field)
                    }
                    (Some(MetadataValue::String(s)), _) => {
                        // Escape single quotes by doubling them
                        let escaped = s.replace('\'', "''");
                        format!("{} {} '{}'", c.field, op_str, escaped)
                    }
                    (Some(MetadataValue::Number(n)), _) => {
                        format!("{} {} {}", c.field, op_str, n)
                    }
                    (Some(MetadataValue::Bool(b)), _) => {
                        format!("{} {} {}", c.field, op_str, b)
                    }
                    (None, _) => String::new(),
                }
            }
            Filter::Group { logic, filters } => {
                let joiner = match logic {
                    FilterLogic::And => " AND ",
                    FilterLogic::Or => " OR ",
                };
                let parts: Vec<String> = filters
                    .iter()
                    .map(|f| f.to_lance_filter())
                    .filter(|s| !s.is_empty())
                    .collect();
                match parts.len() {
                    0 => String::new(),
                    1 => parts.into_iter().next().unwrap(),
                    _ => format!("({})", parts.join(joiner)),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_filter() {
        assert_eq!(Filter::Empty.to_lance_filter(), "");
    }

    #[test]
    fn test_eq_string() {
        let f = Filter::eq("session_id", MetadataValue::String("abc-123".into()));
        assert_eq!(f.to_lance_filter(), "session_id = 'abc-123'");
    }

    #[test]
    fn test_eq_string_with_quotes() {
        let f = Filter::eq("name", MetadataValue::String("it's a test".into()));
        assert_eq!(f.to_lance_filter(), "name = 'it''s a test'");
    }

    #[test]
    fn test_gt_number() {
        let f = Filter::Condition(FilterCondition {
            field: "turn".into(),
            op: FilterOp::Gt,
            value: Some(MetadataValue::Number(5.0)),
        });
        assert_eq!(f.to_lance_filter(), "turn > 5");
    }

    #[test]
    fn test_is_not_null() {
        let f = Filter::is_not_null("project");
        assert_eq!(f.to_lance_filter(), "project IS NOT NULL");
    }

    #[test]
    fn test_and_group() {
        let f = Filter::eq("source", MetadataValue::String("claude".into()))
            .and(Filter::Condition(FilterCondition {
                field: "turn".into(),
                op: FilterOp::Gte,
                value: Some(MetadataValue::Number(3.0)),
            }));
        assert_eq!(
            f.to_lance_filter(),
            "(source = 'claude' AND turn >= 3)"
        );
    }

    #[test]
    fn test_or_group() {
        let f = Filter::eq("source", MetadataValue::String("claude".into()))
            .or(Filter::eq("source", MetadataValue::String("codex".into())));
        assert_eq!(
            f.to_lance_filter(),
            "(source = 'claude' OR source = 'codex')"
        );
    }

    #[test]
    fn test_nested_groups() {
        let inner = Filter::eq("a", MetadataValue::String("1".into()))
            .or(Filter::eq("b", MetadataValue::String("2".into())));
        let outer = inner.and(Filter::eq("c", MetadataValue::Number(3.0)));
        assert_eq!(
            outer.to_lance_filter(),
            "((a = '1' OR b = '2') AND c = 3)"
        );
    }
}
