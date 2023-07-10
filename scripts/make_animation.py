from pymol import cmd

cmd.load("anim.pse")

frame_per_scene = 3
num_scenes = 100
frames_after_last = 100

prev_obj = None
for i in range(num_scenes):
    obj_name = f"pred_150_{i}_5"
    cmd.show("sticks", obj_name)
    if prev_obj is not None:
        cmd.hide("sticks", prev_obj)
    prev_obj = obj_name

    cmd.scene(f"{i}", "store")


cmd.mset(f"1x{frame_per_scene*num_scenes + frames_after_last}")
for i in range(num_scenes):
    cmd.mview("store", i*frame_per_scene, scene=f"{i}")
cmd.mview("store", frames_after_last-1, scene=f"{i}")

cmd.set("ray_trace_frames", 1)
cmd.movie.produce("outputs/anim.mpg", quality=100)