# Event-only update support preflight

Read-only cache audit:55,260 query/history positions,38,431 with known insulin
rate.220 positions have known rate but no observed lab, affecting133queries.
No current query has known rate without observed labs (paired potassium is
required). Positions may repeat between queries, so these are not unique events.

After applying each checkpoint's source-supported lab channels, the affected
fit support is33-42patients/69-94queries; source evaluation5-17patients/8-29
queries. The common hospital group has7patients/10queries. This is below the
existing30patient minimum for subgroup paired inference.

Only~0.57% of known-rate positions are blocked by the current lab-step mask.
This limits a blanket mask-bug explanation; it does not rule out important
effects on those patients or effects on learned weights after retraining.
Do not report unsupported subgroup CIs as failed biology. No efficacy trial
or mask modification was performed on this insufficient subgroup.

Crucial distinction: activating treatment-only positions on the current ten-lag
grid is not native-event replay. Native rates occurring between those positions
are not represented as distinct updates by this cache. A full irregular event
sequence remains a separate experiment; current negative evidence does not
test it. No runtime/calibration/promotion changes.
