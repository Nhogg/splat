# ZED Gaussian Splatting

## Render a trained splat

Pass the Nerfstudio training `config.yml` to the rendering script:

```bash
./render_splat.sh \
  outputs/capture2-zed-depth/splatfacto/2026-07-17_103425/config.yml \
  renders/room.mp4
```

The script loads the associated checkpoint and renders an interpolated camera
path to the specified MP4. Run it from `/home/nhogg/splat` with the
`.venv-ns` environment already installed.

An exported `splat.ply` alone cannot be rendered with this script because
Nerfstudio also needs its training configuration and checkpoint.
