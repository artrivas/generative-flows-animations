# Lab 3 — Generative 2D Visualizer

Visualizador de procesos de difusión (VP/VE/sub-VP) y Flow Matching sobre distribuciones sintéticas 2D.

## Estructura

```
lab3-generative-2d/
├── references/         # Ecuaciones y citas de papers (ver NOTES.md)
├── configs/            # Hydra configs por componente
├── src/                # Implementación
│   ├── data/           # Distribuciones sintéticas 2D
│   ├── forward_process/# SDEs y Flow Matching
│   ├── models/         # Redes (score, velocity)
│   ├── training/       # Loops de entrenamiento
│   ├── sampling/       # Samplers (Euler-Maruyama, DDPM, etc.)
│   ├── integrators/    # ODE integrators
│   └── viz/            # Animaciones y plots
├── scripts/            # Scripts de entrenamiento y visualización
├── checkpoints/        # Modelos guardados
└── outputs/videos/     # Videos generados
```

## Documentación

- [`docs/theory.md`](docs/theory.md) — trasfondo conceptual de cada componente
- [`docs/technical.md`](docs/technical.md) — arquitectura, gráficas y resultados de validación
- [`references/NOTES.md`](references/NOTES.md) — ecuaciones exactas y citas de papers

## Plan de implementación

Pasos 1–13, validando cada uno antes de continuar.

- [x] Paso 1: Distribuciones sintéticas 2D + sanity check visual
- [x] Paso 2: Forward process — VP/VE/sub-VP + kernels analíticos (validado numéricamente, 15/15 OK)
- [x] Paso 3: Animaciones forward_comparison y density_evolution
- [x] Paso 4: Integradores (Euler, Heun, Euler-Maruyama) validados numéricamente
- [x] Paso 5a: Modelo denoising + training loop, prueba de pipeline en CPU (eight_gaussians, ε-pred)
- [x] Paso 5b (preparación): script/notebook de entrenamiento completo listos — **pendiente: correr en Colab/GPU y traer los 8 checkpoints**
- [ ] Paso 6–13: TBD
