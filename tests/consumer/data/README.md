# Frozen robot-profile migration snapshots

`g1_profile.json` and `gr1_profile.json` were generated from the pre-R2b core
commit `7246bfe` (`make_g1_profile` / `make_gr1_profile`) before those factories
were removed. The consumer tests compare every `RobotProfile` field against
these files.

Regenerate them only as an explicit profile-data migration, not from the live
pack loader; regenerating from the loader would make the check tautological.
