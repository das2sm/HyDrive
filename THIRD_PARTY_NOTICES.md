# Third-party notices

HyDrive integrates with and contains files derived from the following
open-source projects:

- [SparseDriveV2](https://github.com/swc-17/SparseDriveV2), licensed under
  Apache-2.0 and used as the unchanged score-and-select planner.
- [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive), providing the
  closed-loop evaluation integration on which SparseDriveV2 is based.
- [Fail2Drive](https://github.com/autonomousvision/fail2drive), licensed under
  MIT and providing the route suite and plugin integration.
- [CARLA](https://github.com/carla-simulator/carla), including Leaderboard and
  ScenarioRunner components.

The exact agent snapshot in `intervention/sparsedrive_b2d_agent_occ.py` is
derived from the SparseDriveV2/Bench2Drive evaluation agent. The patch in
`patches/fail2drive_campaign_controls.patch` modifies the Fail2Drive plugin
integration.
Redistributed upstream-derived files retain their upstream terms and notices.

HyDrive changes to the derived agent add privileged occupancy construction,
proposal filtering and reselection, compact campaign logging, runtime checks,
and the multi-seed evaluation configuration. These modifications are
identified in the source file as required by Apache-2.0.

Applicable license texts included with this release are stored in `licenses/`.
The top-level MIT license applies to original HyDrive code and documentation;
it does not replace third-party licenses.
