# Parkbox2

Real-time plant growth simulation paired with a Vulkan rendering engine specialized for dense foliage.

PBX2 leverages several modern GPU features and techniques to achieve high performance:
- Instanced rendering
- Indirect GPU-driven rendering
- Multi-draw
- Bindless rendering
- Clustered forward rendering
- Depth pyramid culling

All these rendering techniques accelerate modern visuals:
- PBR rendering (with IBL)
- Ground-truth AO
- Temporal soft shadows (no PCF, PCSS, or VSMs, etc).

## Graphics

The PBR rendering is primarily based on Filament.

Ground-truth AO is implemented based on the original paper.

Temporal soft shadows are a niche shadow filtering technique. I only know of one other renderer which does this; Blender Eevee.
They are still a quite performant solution to shadow filtering as they give you PCSS-like contact hardening whilst only requring a single sample from the shadow map.
The temporal accumulation currently only reprojects UV and rejects based on depth. Motion vectors are not used (yet).
