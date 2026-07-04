# Teoría — Lab 3: Visualizador de Procesos Generativos en 2D

Documento de acompañamiento conceptual. Las ecuaciones exactas, citas y números
de ecuación viven en [`references/NOTES.md`](../references/NOTES.md); aquí se
explica el *por qué* de cada pieza y cómo encajan entre sí.

---

## 1. Idea central: destruir datos para aprender a reconstruirlos

Los modelos generativos basados en score (score-based) y de difusión parten de
una observación simple: es mucho más fácil especificar un **proceso que
destruye** una distribución de datos $p_{\text{data}}$ hasta convertirla en
ruido puro $\mathcal{N}(0, I)$, que especificar directamente cómo generar
datos desde cero.

Ese proceso de destrucción —el **forward process**— se define como una
ecuación diferencial estocástica (SDE):

$$
d\mathbf{x} = f(\mathbf{x}, t)\,dt + g(t)\,d\mathbf{w}, \qquad t \in [0, 1]
$$

con $\mathbf{x}_0 \sim p_{\text{data}}$ y $\mathbf{x}_1 \approx \mathcal{N}(0,I)$.
Una vez fijado el forward process, existe (Anderson, 1982; formalizado en
Song et al. 2021) un proceso reverso que, si se conoce el *score*
$\nabla_{\mathbf{x}}\log p_t(\mathbf{x})$ en cada instante $t$, permite
recorrer el camino inverso: de ruido a datos. Ese score es exactamente lo que
una red neuronal aprende a aproximar durante el entrenamiento.

Este laboratorio visualiza la mitad "fácil" del problema —el forward
process— en 2D, donde cada paso intermedio se puede dibujar directamente.

---

## 2. Las distribuciones sintéticas (paso 1)

Para poder *ver* cómo cada proceso forward destruye estructura, se necesitan
distribuciones 2D con geometrías cualitativamente distintas. Se implementaron
cinco, cubriendo cuatro categorías pedidas:

| Categoría | Distribución | Por qué importa |
|---|---|---|
| Multimodal | `eight_gaussians` | Expone cómo un proceso funde modos separados en una sola masa gaussiana |
| Soporte curvo | `two_moons` | La curvatura del soporte se pierde progresivamente bajo ruido isotrópico |
| Regiones desconectadas | `checkerboard` | Los bordes discretos (discontinuidades de densidad) son el caso más difícil para cualquier proceso de suavizado |
| Geometría compleja | `pinwheel` | Combina rotación + escala radial; buen stress-test para la anisotropía del ruido |
| (bonus) enredo topológico | `two_spirals` | Dos componentes conexas entrelazadas, distinto de "two moons" (que son componentes separadas) |

Ver [`outputs/sanity_checks/overview.png`](../outputs/sanity_checks/overview.png)
para una comparación visual con ejes idénticos (±5.5 en ambos ejes), y la
sección 3 de [`docs/technical.md`](technical.md) para el detalle de
implementación y las verificaciones numéricas hechas sobre estas muestras.

---

## 3. Los tres procesos forward (paso 2)

Todos comparten la forma general $d\mathbf{x} = f(\mathbf{x},t)dt + g(t)d\mathbf{w}$,
pero difieren en cómo escalan la señal original ($\mathbf{x}_0$) y cuánta
varianza acumulan con el tiempo. La intuición de cada uno:

### VP (Variance Preserving)

$$
d\mathbf{x} = -\tfrac{1}{2}\beta(t)\mathbf{x}\,dt + \sqrt{\beta(t)}\,d\mathbf{w}
$$

El drift **encoge** $\mathbf{x}$ hacia el origen exactamente al mismo ritmo
que la difusión inyecta ruido, de modo que la **varianza total del proceso
se mantiene acotada en 1** para todo $t$ (de ahí el nombre). Es la SDE
continua subyacente a DDPM (Ho et al. 2020).

### VE (Variance Exploding)

$$
d\mathbf{x} = \sqrt{\tfrac{d\sigma^2(t)}{dt}}\,d\mathbf{w}
$$

No hay drift: $\mathbf{x}_0$ nunca se encoge, solo se le suma ruido de
varianza creciente $\sigma(t)^2$ sin cota superior — de ahí "exploding". Es
la SDE continua subyacente a SMLD/NCSN (Song & Ermon, 2019).

### sub-VP (Sub-Variance Preserving)

$$
d\mathbf{x} = -\tfrac{1}{2}\beta(t)\mathbf{x}\,dt + \sqrt{\beta(t)\left(1-e^{-2\int_0^t\beta(s)ds}\right)}\,d\mathbf{w}
$$

Mismo drift que VP, pero con un coeficiente de difusión más pequeño,
diseñado para que la varianza acumulada esté **estrictamente por debajo**
de la del VP en todo $t>0$ (de ahí "sub"). En la práctica produce
trayectorias visualmente "más suaves" que el VP puro.

### Por qué necesitamos el kernel cerrado

Simular cualquiera de estas SDEs paso a paso (Euler-Maruyama) es costoso y
acumula error de discretización. Pero como las tres son SDEs **lineales**
(el drift es lineal en $\mathbf{x}$ y la difusión no depende de $\mathbf{x}$),
existe una solución cerrada para la distribución marginal
$q(\mathbf{x}_t\mid\mathbf{x}_0)$: es gaussiana, con media y varianza
calculables analíticamente sin integrar nada numéricamente. Este kernel
cerrado es lo que permite entrenar modelos de difusión eficientemente
(muestrear $\mathbf{x}_t$ directamente para cualquier $t$, sin recorrer los
pasos intermedios) y es también la pieza que se validó numéricamente en
este paso — ver tabla de resultados en
[`docs/technical.md`](technical.md#validación-numérica-del-kernel-cerrado-paso-2).

Nota sobre el kernel de sub-VP: a diferencia de VP y VE, **no estaba
documentado explícitamente en `references/NOTES.md`** con un número de
ecuación de la fuente original. Se derivó analíticamente a partir de la
solución general de SDEs lineales (isometría de Itô) — ver
`src/forward_process/kernels.py` para la derivación completa — y se validó
por Monte Carlo. Queda pendiente contrastarlo contra el Apéndice B del
paper de Song et al. 2021 por si aparece ahí en forma cerrada.

---

## 4. Qué significa "validar" el kernel cerrado

La validación de este paso no vuelve a resolver la SDE por integración
numérica (eso se reserva para un paso posterior de comparación
solver-vs-forma-cerrada). En su lugar, comprueba que la implementación de
`marginal_prob()` — shapes de tensores, broadcasting, aritmética — es
internamente consistente: si se generan 50 000 muestras vía
$\mathbf{x}_t = \mu_t(\mathbf{x}_0) + \sigma_t \cdot \boldsymbol{\varepsilon}$
usando los valores que la propia función reporta, la media y desviación
estándar empíricas de esas muestras deben coincidir con $\mu_t,\sigma_t$
dentro del margen de ruido de Monte Carlo.

Un detalle importante descubierto durante la validación: cuando
$\alpha(t)\to 0$ (t cercano a 1 en VP/sub-VP), la media predicha
$\mu_t=\alpha(t)\mathbf{x}_0$ se acerca a cero por diseño — el proceso está
borrando la señal original, que es justamente su propósito. Medir el error
relativo dividiendo entre esa media casi nula infla artificialmente el
error aunque la implementación sea correcta. La métrica correcta normaliza
el error absoluto contra una escala fija ($\|\mathbf{x}_0\|$), no contra la
cantidad que se está midiendo. Ver la tabla completa y la discusión en
[`docs/technical.md`](technical.md).
