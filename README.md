# generative-flows-animations

Laboratorio de visualización de modelos generativos 2D (SDEs VP/VE/sub-VP,
denoising score matching con parametrización ε/v, reverse-SDE, Probability
Flow ODE, y Flow Matching) sobre distribuciones sintéticas 2D.

El proyecto vive en [`lab3-generative-2d/`](lab3-generative-2d/). La
documentación está dividida en tres documentos, cada uno con un propósito
distinto:

| Documento | Contenido |
|---|---|
| [`lab3-generative-2d/README.md`](lab3-generative-2d/README.md) | **Manual de uso**: instalación, punto de entrada único (`scripts/main.py train\|sample\|animate`), cómo entrenar, cargar checkpoints existentes, generar muestras y regenerar cada animación cambiando distribución/modelo/sampler/pasos/semilla por flags de línea de comandos. |
| [`lab3-generative-2d/docs/theory.md`](lab3-generative-2d/docs/theory.md) | **Documentación teórica**: trasfondo conceptual de los procesos de difusión y Flow Matching implementados. |
| [`lab3-generative-2d/docs/technical.md`](lab3-generative-2d/docs/technical.md) | **Documentación técnica / reporte de auditoría**: estructura del proyecto, decisiones de implementación, hallazgos de revisión de código, verificaciones (ej. óptimo de Bayes para `eight_gaussians`), limitaciones conocidas (ej. `pinwheel`) y su estado. |

Las ecuaciones y citas exactas de cada componente están en
[`lab3-generative-2d/references/NOTES.md`](lab3-generative-2d/references/NOTES.md).
