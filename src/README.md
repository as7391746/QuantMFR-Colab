Snapshot of the RiskUncertaintyValue expansion engine
(github.com/lphansen/RiskUncertaintyValue, branch Planners_with_External,
commit 09ca5df), unmodified — the same src/ layout as the upstream
repository, so `sys.path.insert(0, "<repo>/src")` followed by
`from uncertain_expansion import uncertain_expansion` works exactly as in
the book appendix. expansion/expansion.py is the user-facing layer over it.
