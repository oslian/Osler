# Conditional Owner-State Readout Pilot

Fixed seed7, exact original64fit32tune64evaluation64hospital selection and
eight-epoch optimizer/loss. One shared trajectory, identity graph, closed body
bypass, owner-current state and future change feed value/JEPA zero-start heads.
Three arms: conditional (both states), delta-only (current state zeroed),
anchor-only (both state blocks zeroed). Same initialization and parameter shapes;
effective capacity differs deliberately. Time/anchor/mask present in all arms.
Source tune selects epoch0..8; previously inspected held-out roles diagnostic
only. No promotion, calibration or runtime changes. Original delta-only bound
module remains an experimental reference, not an added serving compatibility layer.
