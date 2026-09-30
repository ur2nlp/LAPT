# `lapt_core.fetch_diff`

Compare a remote run inventory against local files and emit the `scp` commands
for what is missing or stale.

A project supplies its environment-variable prefix — `LAPT` gives `$LAPT_REMOTE`
and `$LAPT_OUTPUTS_DIR` — and whether to zero-pad `v`-prefixed experiment ids.
Padding is off by default; turn it on only where ids are typed by hand and the
same run can be written more than one way.

Needs no extra — standard library plus PyYAML.

::: lapt_core.fetch_diff
