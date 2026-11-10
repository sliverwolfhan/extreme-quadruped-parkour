# Videos Notes

## Hurdle clip variants
There are two hurdle clips because the default hurdle generator increases height
and tightens the gap as the level goes up. At level 9, the hurdles are too close,
so the robot collides with the hurdles frequently. Therefore:

- Level 8 hurdle clip: recorded with the standard play command.
- Level 9 hurdle clip: recorded with a fixed, wider gap (1.4 m) and max level 9.

All other terrain clips are level 9 recordings produced with the standard play command.

### Level 8 (default hurdle gap progression - standard play command)
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Go2-Rough-Direct-Abl3_5-v0 \
  --num_envs 50 \
  --load_run <run_dir_name> \
  --checkpoint <checkpoint_file>
```

### Level 9 (fixed hurdle gap = 1.4 m)
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Go2-Rough-Direct-Abl3_5-v0 \
  --num_envs 50 \
  --checkpoint <checkpoint_file> \
  env.terrain.terrain_generator.sub_terrains.hurdle_strip.hurdle_gap_range="[1.4, 1.4]" \
  env.terrain.max_init_terrain_level=9
```
