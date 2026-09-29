# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "diffusers",
#     "einops",
#     "marimo",
#     "matplotlib",
#     "torch",
# ]
# ///

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To run in molab: Select Server compute and GPU: RTX Pro 6000 Blackwell. Then run the notebook with the play button

    For a better experience, prefer using appview (toggle button under save button)

    You pull the repo locally from [github](https://github.com/gatienc/discrete-flow-matching-notebook/tree/master)

    Feedback and potential PR very appreciated! 😎
    """)
    return


@app.cell(hide_code=True, expand_output=True)
def header(
    COLOR_NAMES,
    PALETTE,
    SHAPE_NAMES,
    device,
    draw_cycle,
    draw_item,
    mo,
    plt,
    randomize_button,
    sample_pair,
    to_rgb,
):
    _shape_cycle_figure = draw_cycle(
        "Shape rule: the shape morphs",
        SHAPE_NAMES,
        lambda axis, index, center: draw_item(axis, SHAPE_NAMES[index], center, "0.8"),
    )
    _color_cycle_figure = draw_cycle(
        "Color rule: the color cycles",
        COLOR_NAMES,
        lambda axis, index, center: draw_item(axis, "circle", center, PALETTE[1 + index].tolist()),
    )

    # example pair: re-drawn whenever the randomize button is clicked
    _randomize_clicks = randomize_button.value
    _x0, _x1 = sample_pair(1)
    x0_preview, x1_preview = _x0.to(device), _x1.to(device)
    _example_figure, _example_axes = plt.subplots(1, 2, figsize=(6.0, 2.9))
    for _axis, _img, _title in zip(
        _example_axes, (x0_preview[0], x1_preview[0]), ("input", "target"), strict=True
    ):
        _axis.imshow(to_rgb(_img))
        _axis.set_title(_title)
        _axis.set_xticks(())
        _axis.set_yticks(())
        for _spine in _axis.spines.values():
            _spine.set_edgecolor("black")
            _spine.set_linewidth(1.2)
    _example_figure.tight_layout(w_pad=4)
    _example_figure.text(0.5, 0.45, "→", ha="center", va="center", fontsize=26)

    mo.vstack(
        [
            mo.md(r"""
    # Discrete flow matching

    This notebook implements a toy example of [Discrete Flow Matching (Gat et al. 2024)](http://arxiv.org/abs/2407.15595), using the notation introduced in the paper. We recommend reading the paper to understand the mathematical foundations of the approach. This notebook only focuses on the implementation in a simple 2D case. We expect the reader to already have a basic understanding of continuous flow matching.

    This notebook was heavily inspired by Georges Le Bellier's [educational notebook](https://github.com/lebellig/discrete-fm/tree/master) on discrete flow matching.

    If you don't care about the details, just scroll down and train your own model! It trains in a few seconds with 1k steps (the random coupling needs more steps for good results).

    ## Dataset
    To get an understanding of discrete flow matching behaviour, we algorithmically generate a toy "dataset" of random non-overlapping items (squares/circles/triangles in red/blue/green) whose **shape and color** morph to the next ones along two independent cycles (represented here).
    """),
            mo.hstack([_shape_cycle_figure, _color_cycle_figure], justify="center"),
            mo.hstack([_example_figure], justify="center"),
            mo.hstack([randomize_button], justify="center"),
        ]
    )
    return x0_preview, x1_preview


@app.cell(hide_code=True)
def path_explanation(mo, model_architecture):
    mo.vstack(
        [
            mo.md(r"""
    ## Model
    A small time-conditioned U-Net (diffusers `UNet2DModel`) takes the current state $x_t$ (one-hot over background / red / blue / green) and the time $t$, and predicts for each pixel the probability of each class at the endpoint, $p_{1|t}(x_1^i \mid x_t)$.
    """),
            model_architecture,
            mo.md(r"""
    ## What is a sample
    To keep the explanation simple we consider the linear case:
    In continuous flow matching:
    $x_t= x_0(1-t) + x_1 \cdot t$

    with $x_0$ and $x_1$ a matched pair from the input and output distributions.
    However in discrete flow matching it is not possible to continuously linearly interpolate between 2 samples. So we sample them instead!

    $$x_t^i = \begin{cases} x_1^i & \text{with probability } t \\ x_0^i & \text{with probability } 1 - t \end{cases}$$

    (with $i$ the pixel index)

    Each pixel of $x_t$ is the corresponding pixel of the target $x_1$ with probability $t$, and the pixel of the source $x_0$ otherwise. Here is a visual explanation of how the samples are created during training:
    """),
        ]
    )
    return


@app.cell
def _(mo, path_t, plot_path_sample):
    # fresh uniform draw on every run: no link to the previous slider position
    mo.vstack([path_t, plot_path_sample(path_t.value)])
    return


@app.cell(hide_code=True)
def training_explanation(mo):
    mo.md(r"""
    ## Training
    Where continuous flow matching regresses a velocity, here the model classifies each pixel. One training step:

    1. draw a pair $(x_0, x_1)$ and a time $t \sim \mathcal{U}(0, 1)$
    2. sample $x_t$ as above: each pixel comes from $x_1$ with probability $t$, from $x_0$ otherwise
    3. the model predicts, for each pixel, the probability of each class of $x_1$
    4. minimize the cross-entropy against the true $x_1$:

    $$\mathcal{L} = -\frac{1}{N} \sum_i \log p_{1|t}(x_1^i \mid x_t)$$

    (with $N$ the number of pixels)

    The model is **unconditional**: it only sees $x_t$ and $t$, never $x_0$. The source only matters as the starting point of the flow, which is what the coupling dropdown below changes: the source $x_0$ or random noise.
    """)
    return


@app.cell(hide_code=True)
def sampling_explanation(mo):
    mo.md(r"""
    ## Inference Sampling
    Inference sampling is a bit different from continuous flow matching. With the discrete equivalent of an Euler sampler, at each step from $t$ to $t + h$ the model predicts, for each pixel, the probability of each class of the endpoint. The sampler draws a guess from these probabilities, then lets every pixel jump to its guess with a probability of $\frac{\kappa_{t+h} - \kappa_t}{1 - \kappa_t}$ ($\frac{h}{1 - t}$ for the linear scheduler), and keep its current class otherwise.
    """)
    return


@app.cell
def jump_example(
    PALETTE,
    guess_resample_button,
    jump_resample_button,
    mo,
    torch,
):
    # one background pixel, one Euler step (linear scheduler)
    # each draw is seeded by its own button's click count, so a click only re-draws its step
    # (even seeds for the guess, odd seeds for the jump: independent streams)
    _p1 = {"blue": 0.5, "background": 0.4, "red": 0.1}
    _t, _h = 0.5, 0.25
    _weight = _h / (1 - _t)

    _guess_generator = torch.Generator().manual_seed(2 * guess_resample_button.value)
    _guess = list(_p1)[
        torch.multinomial(torch.tensor(list(_p1.values())), 1, generator=_guess_generator).item()
    ]
    _jump_generator = torch.Generator().manual_seed(2 * jump_resample_button.value + 1)
    _u = torch.rand((), generator=_jump_generator).item()
    _kept = _u < _weight
    _new = _guess if _kept else "background"

    def _css_rgb(name):
        _r, _g, _b = (PALETTE[{"background": 0, "red": 1, "blue": 2}[name]] * 255).int().tolist()
        return f"rgb({_r},{_g},{_b})"

    def _pixel(name):
        return (
            '<span style="display:inline-block;width:1.6em;height:1.6em;vertical-align:middle;'
            f'background:{_css_rgb(name)};border:1.5px solid black"></span>'
        )

    _segments = "".join(
        f'<span style="display:inline-block;width:{_p * 100:.0f}%;padding:0.2em 0;text-align:center;'
        f"background:{_css_rgb(_name)};color:{'black' if _name == 'background' else 'white'};"
        f"font-size:0.8em;outline:{'3px solid black' if _name == _guess else 'none'};"
        f'outline-offset:-3px">{_name} {_p:.0%}</span>'
        for _name, _p in _p1.items()
    )
    _bar = (
        '<span style="display:inline-flex;width:18em;vertical-align:middle;'
        f'border:1.5px solid black">{_segments}</span>'
    )

    _row_guess = (
        f"**1. Sample from the model prediction** &nbsp; current pixel {_pixel('background')}"
        f" &nbsp;→&nbsp; {_bar} &nbsp;→&nbsp; guess {_pixel(_guess)} *{_guess}*"
    )
    _row_jump = (
        "**2. Keep the model prediction?** &nbsp; at $t = "
        + f"{_t}$, step $h = {_h}$: "
        + r"$\frac{h}{1 - t} = "
        + f"{_weight:.2f}$"
        + f", draw $u = {_u:.2f}$ "
        + (r"$<$" if _kept else r"$\geq$")
        + f" {_weight:.2f} &nbsp;→&nbsp; "
        + ("keep the guess" if _kept else "keep the current pixel")
        + f" &nbsp;→&nbsp; new pixel {_pixel(_new)} *{_new}*"
    )

    mo.vstack(
        [
            mo.md(
                "**Example.** One background pixel; the model predicts blue 50%, "
                "background 40% and red 10%."
            ),
            mo.hstack([mo.md(_row_guess), guess_resample_button], justify="start", align="center"),
            mo.hstack([mo.md(_row_jump), jump_resample_button], justify="start", align="center"),
            mo.md(
                "Predicting the current class (background) is also a *no change*: over many draws "
                "the pixel ends up blue 25%, red 5% and background 70% ($0.5 + 0.5 \\times 0.4$)."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def train_controls(coupling_choice, kappa_choice, mo, train_steps_choice):
    mo.vstack(
        [
            mo.md(r"""
    ## Train your own model
    Pick a coupling, a scheduler and a number of training steps, then click Train.
    """),
            mo.hstack([coupling_choice, kappa_choice, train_steps_choice], justify="start"),
        ]
    )
    return


@app.cell(hide_code=True)
def cubic_scheduler_plot(
    CubicScheduler,
    LinearScheduler,
    a_choice,
    b_choice,
    kappa_choice,
    mo,
    plt,
    torch,
):
    mo.stop(kappa_choice.value != "cubic")

    _grid = torch.linspace(0, 1, 200)
    _schedulers = {
        "Linear": LinearScheduler(),
        "Cubic": CubicScheduler(a_choice.value, b_choice.value),
    }

    _fig, _axes = plt.subplots(1, 2, figsize=(9.5, 3.2))
    for _name, _sched in _schedulers.items():
        _axes[0].plot(_grid, _sched(_grid), label=_name)
        _axes[1].plot(_grid, _sched.derivative(_grid), label=_name)

    _axes[0].set_title("κ(t): mass at x₁ by time t")
    _axes[0].set_xlabel("t")
    _axes[0].set_ylabel("κ(t)")
    _axes[1].set_title("κ′(t): rate of mass transport")
    _axes[1].set_xlabel("t")
    _axes[1].set_ylabel("κ′(t)")
    for _axis in _axes:
        _axis.axhline(0, color="gray", lw=0.5)
        _axis.legend()
        _axis.grid(alpha=0.3)
    _fig.suptitle(
        f"Cubic scheduler κ(t) = -2t³ + 3t² + a(t³ - 2t² + t) + b(t³ - t²), "
        f"a={a_choice.value}, b={b_choice.value}",
        fontsize=9,
    )
    _fig.tight_layout()
    mo.vstack(
        [
            mo.md(r"""
    In the paper, $a = 0, b = 2$ ($\kappa_t = t^2$) gives the best text generation and is used for all their language models.
    """),
            a_choice,
            b_choice,
            _fig,
        ]
    )
    return


@app.cell(hide_code=True)
def wiring(
    CubicScheduler,
    DiscreteFM,
    LinearScheduler,
    RandomCoupling,
    X0Coupling,
    copy,
    device,
    dict_size,
    mo,
    model_architecture,
    sample_pair,
    torch,
    train_button,
):
    _settings = train_button.value
    mo.stop(_settings is None, train_button)
    mo.output.replace(train_button)

    _coupling = {
        "x0": X0Coupling,
        "random": RandomCoupling,
    }[_settings["coupling"]]
    _coupling = _coupling(dict_size) if _settings["coupling"] == "random" else _coupling()
    _kappa = (
        LinearScheduler()
        if _settings["kappa"] == "linear"
        else CubicScheduler(_settings["a"], _settings["b"])
    )

    trained_model = DiscreteFM(dict_size, copy.deepcopy(model_architecture), _coupling, _kappa).to(
        device
    )
    _optimizer = torch.optim.Adam(trained_model.parameters(), lr=3e-3)

    _losses = []
    trained_model.train()
    for _step in mo.status.progress_bar(
        range(10 ** _settings["steps"]),
        title="Training",
        completion_title="Training done",
        show_rate=True,
        show_eta=True,
    ):
        _x0, _x1 = sample_pair(8)
        _loss = trained_model.step((_x0.to(device), _x1.to(device)))
        _optimizer.zero_grad()
        _loss.backward()
        _optimizer.step()
        _losses.append(_loss.item())

    selected_coupling_name = {
        "x0": "x0 start",
        "random": "random (uniform noise)",
    }[_settings["coupling"]]
    _kappa_label = {"linear": "Linear", "cubic": "Cubic scheduler"}[_settings["kappa"]]
    selected_kappa_name = _kappa_label + (
        f" (a={_settings['a']}, b={_settings['b']})" if _settings["kappa"] == "cubic" else ""
    )
    print(
        f"{selected_coupling_name} | {selected_kappa_name} | device={device} | "
        f"loss: {_losses[0]:.4f} → {_losses[-1]:.6f}"
    )
    return selected_coupling_name, selected_kappa_name, trained_model


@app.cell(hide_code=True)
def sampling(
    dict_size,
    discrete_euler_sample,
    inference_step,
    selected_coupling_name,
    selected_kappa_name,
    trained_model,
    x0_preview,
    x1_preview,
    x2prob,
):
    sample_steps = inference_step.value
    shown_target = x1_preview

    # the flow start must match the training coupling
    sample_start, _ = trained_model.coupling.sample(x0_preview, shown_target)
    _initial_state = x2prob(sample_start, dict_size).float()

    sample_trajectory, sample_times = [], []

    def _record_state(state, time):
        sample_trajectory.append(state.detach().clone())
        sample_times.append(float(time))

    def _model_fn(state, time):
        return trained_model(time, state)

    # multinomial proposals with mixture-weight rounding at every step
    _final_state = discrete_euler_sample(
        _model_fn,
        _initial_state,
        steps=sample_steps,
        kappa=trained_model.kappa,
        on_state=_record_state,
    )
    sample_has_target = trained_model.coupling.has_fixed_target
    if sample_has_target:
        sample_accuracy = (_final_state.argmax(dim=1) == shown_target).float().mean().item()
        print(
            f"{selected_coupling_name} | {selected_kappa_name} | "
            f"{sample_steps}-step Euler (multinomial sampler) accuracy: {sample_accuracy:.3f}"
        )
    else:
        # generation from noise: no reference target, accuracy is meaningless
        sample_accuracy = None
        print(
            f"{selected_coupling_name} | {selected_kappa_name} | "
            f"{sample_steps}-step Euler (multinomial sampler), "
            "unconditional generation from noise"
        )
    return (
        sample_accuracy,
        sample_has_target,
        sample_start,
        sample_steps,
        sample_times,
        sample_trajectory,
        shown_target,
    )


@app.cell(hide_code=True)
def trajectory_controls(
    inference_step,
    mo,
    randomize_button,
    sample_iteration,
    show_model_prediction,
    show_soft_map,
):
    mo.vstack(
        [
            mo.md(r"""
    ## Results
    The trained model samples the example pair above. The trajectory slider scrubs through the recorded Euler states, and the error map shows wrong pixels in their target color. *Show model endpoint prediction p₁* replaces the state by the model prediction of $x_1$ at that time, and the soft view blends the palette by class probability.
    """),
            inference_step,
            sample_iteration,
            show_model_prediction,
            show_soft_map,
            randomize_button,
        ]
    )
    return


@app.cell(hide_code=True)
def figure(
    plt,
    probs_to_rgb,
    sample_accuracy,
    sample_has_target,
    sample_iteration,
    sample_start,
    sample_steps,
    sample_times,
    sample_trajectory,
    show_model_prediction,
    show_soft_map,
    shown_target,
    to_rgb,
    torch,
    trained_model,
):
    _iteration = sample_iteration.value
    _selected_state = sample_trajectory[_iteration]  # (n, C, H, W)
    _selected_time = float(sample_times[_iteration])
    if show_model_prediction.value:
        _model_was_training = trained_model.training
        trained_model.eval()
        _time_batch = torch.full(
            (_selected_state.shape[0],), _selected_time, device=_selected_state.device
        )
        with torch.inference_mode():
            _probs = torch.softmax(trained_model(_time_batch, _selected_state), dim=1)
        trained_model.train(_model_was_training)
        _display_origin = "model prediction p₁"
    else:
        _probs = _selected_state
        _display_origin = "trajectory state"

    _argmax = _probs.argmax(dim=1)  # (n, H, W)
    _frames = probs_to_rgb(_probs) if show_soft_map.value else to_rgb(_argmax)

    # the sampler records: state 0 = flow start (coupling), then one state per Euler step
    _readout = "flow start (coupling)" if _iteration == 0 else "multinomial proposal"

    if sample_has_target:
        # error map in the same palette: white where correct, target tone where wrong
        _error_rgb = to_rgb(torch.where(_argmax == shown_target, 0, shown_target))
        _fig, _axes = plt.subplots(1, 4, figsize=(10.5, 2.8))
        _panels = [
            (to_rgb(sample_start[0]), "flow start x₀ (coupling)"),
            (to_rgb(shown_target[0]), "target x₁"),
            (
                _frames[0],
                f"{_display_origin} · t={_selected_time:.3f} · {_readout}"
                + (" · soft view" if show_soft_map.value else " · hard classes"),
            ),
            (_error_rgb[0], "error (target tone = wrong pixel)"),
        ]
        _title_suffix = f"accuracy {sample_accuracy:.3f} | "
    else:
        # generation from noise: no reference target, no error map
        _fig, _axes = plt.subplots(1, 2, figsize=(5.6, 2.8))
        _panels = [
            (to_rgb(sample_start[0]), "random start x₀ (uniform noise)"),
            (
                _frames[0],
                f"current sample · t={_selected_time:.3f} · {_readout}"
                + (" · soft view" if show_soft_map.value else " · hard classes"),
            ),
        ]
        _title_suffix = ""
    for _axis, (_img, _title) in zip(_axes, _panels, strict=True):
        _axis.imshow(_img.detach().cpu())
        _axis.set_title(_title, fontsize=9)
        _axis.set_xticks(())
        _axis.set_yticks(())
        for _spine in _axis.spines.values():
            _spine.set_visible(True)
            _spine.set_edgecolor("black")
            _spine.set_linewidth(1.2)
    _fig.suptitle(
        f"{sample_steps}-step Euler | {_title_suffix}state {_iteration}/{sample_steps}",
        fontsize=9,
    )
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def under_the_hood(mo):
    mo.md(r"""
    ## Under the hood
    The cells below implement discrete flow matching: probability helpers, couplings,
    κ schedulers, the Euler sampler and the model. Their code is hidden: expand a cell to read it.
    Marimo UI definitions are last.
    """)
    return


@app.cell(hide_code=True)
def _(rearrange, torch):
    Prob = torch.Tensor
    Img = torch.Tensor

    def x2prob(x: Img, dict_size: int) -> Prob:
        x = torch.nn.functional.one_hot(x, num_classes=dict_size)
        return rearrange(x, "b h w c -> b c h w")

    def sample_p(pt: Prob) -> Img:
        b, _, h, w = pt.shape
        pt = rearrange(pt, "b c h w -> (b h w) c")
        xt = torch.multinomial(pt, 1)
        return xt.reshape(b, h, w)

    return Img, Prob, sample_p, x2prob


@app.cell(hide_code=True)
def _(Img, KappaScheduler, Prob, sample_p, torch):
    def sample_cond_pt(p0: Prob, p1: Prob, t: torch.Tensor | float, kappa: KappaScheduler) -> Img:
        t = t.reshape(-1, 1, 1, 1)
        pt = (1 - kappa(t)) * p0 + kappa(t) * p1
        return sample_p(pt)

    return (sample_cond_pt,)


@app.cell(hide_code=True)
def _(Img, torch):
    class Coupling:
        def sample(self, x0: Img, x1: Img) -> tuple[Img, Img]:
            raise NotImplementedError

    class X0Coupling(Coupling):
        """Start the flow from the source image x0 itself."""

        has_fixed_target = True

        def sample(self, x0: Img, x1: Img) -> tuple[Img, Img]:
            return x0, x1

    class RandomCoupling(Coupling):
        """Random start: independent uniform classes per pixel (discrete noise).

        With this start the model generates a plausible item layout from noise,
        i.e. it samples from the marginal p(x1).
        """

        has_fixed_target = False  # generation from noise: no reference target

        def __init__(self, dict_size: int) -> None:
            self.dict_size = dict_size

        def sample(self, x0: Img, x1: Img) -> tuple[Img, Img]:
            noise = torch.randint(self.dict_size, tuple(x1.shape), device=x1.device)
            return noise, x1

    return Coupling, RandomCoupling, X0Coupling


@app.cell(hide_code=True)
def _(torch):
    class KappaScheduler:
        def __call__(self, t: float | torch.Tensor) -> float | torch.Tensor:
            raise NotImplementedError

        def derivative(self, t: float | torch.Tensor) -> float | torch.Tensor:
            raise NotImplementedError

    class CubicScheduler(KappaScheduler):
        def __init__(self, a: float = 2.0, b: float = 0.5) -> None:
            self.a = a
            self.b = b

        def __call__(self, t: float | torch.Tensor) -> float | torch.Tensor:
            return (
                -2 * (t**3) + 3 * (t**2) + self.a * (t**3 - 2 * t**2 + t) + self.b * (t**3 - t**2)
            )

        def derivative(self, t: float | torch.Tensor) -> float | torch.Tensor:
            return (
                -6 * (t**2) + 6 * t + self.a * (3 * t**2 - 4 * t + 1) + self.b * (3 * t**2 - 2 * t)
            )

    class LinearScheduler(KappaScheduler):
        def __call__(self, t: float | torch.Tensor) -> float | torch.Tensor:
            return t

        def derivative(self, t: float | torch.Tensor) -> float | torch.Tensor:
            return torch.ones_like(t) if isinstance(t, torch.Tensor) else 1.0

    return CubicScheduler, KappaScheduler, LinearScheduler


@app.cell(hide_code=True)
def _(F, torch):
    @torch.inference_mode()
    def discrete_euler_sample(model_fn, initial, *, steps, kappa, on_state=None):
        """Sample the discrete flow with categorical Euler updates.

        ``model_fn(state, t)`` returns endpoint class logits for a one-hot state
        ``(b, c, h, w)``. Each step from ``t`` to ``t_next`` draws a multinomial proposal
        from the predicted endpoint and switches each pixel to it with the mixture weight
        ``(kappa(t_next) - kappa(t)) / (1 - kappa(t))``. The last step has unit weight, so
        every pixel takes its drawn class. ``on_state(state, t)`` records every state.
        """
        times = torch.linspace(0.0, 1.0, steps + 1, device=initial.device)
        num_classes = initial.shape[1]
        state = initial.float().clone()
        if on_state is not None:
            on_state(state, times[0])
        for _index, _start in enumerate(times[:-1]):
            _end = times[_index + 1]
            _probs = torch.softmax(model_fn(state, _start.expand(state.shape[0])).float(), dim=1)
            _mixture_weight = (kappa(_end) - kappa(_start)) / (1.0 - kappa(_start))
            _flat = _probs.movedim(1, -1).reshape(-1, num_classes)
            _classes = torch.multinomial(_flat, 1).reshape(state.shape[:1] + state.shape[-2:])
            _proposal = F.one_hot(_classes, num_classes).movedim(-1, 1).float()
            _replace = torch.rand(_classes.shape, device=state.device) < _mixture_weight
            state = torch.where(_replace[:, None], _proposal, state)
            if on_state is not None:
                on_state(state, _end)
        return state

    return (discrete_euler_sample,)


@app.cell(hide_code=True)
def _(
    Coupling,
    F,
    FlowUNet,
    Img,
    KappaScheduler,
    nn,
    sample_cond_pt,
    torch,
    x2prob,
):
    class DiscreteBackbone(nn.Module):
        """Wrap FlowUNet so ``backbone(t, x)`` works.

        ``x`` is either a discrete class image ``(b, h, w)`` (one-hot encoded) or an
        already-one-hot / soft probability state ``(b, c, h, w)``. Unconditional:
        the backbone sees only the flow state, never the source x0.
        """

        def __init__(self, dict_size: int, base: int = 8, num_blocks: int = 2) -> None:
            super().__init__()
            self.dict_size = dict_size
            self.flow = FlowUNet(channels=dict_size, base=base, num_blocks=num_blocks)

        def forward(self, t, x):
            state = x2prob(x, self.dict_size).float() if x.dim() == 3 else x.float()
            return self.flow(state, t.float())

    class DiscreteFM(nn.Module):
        def __init__(
            self,
            dict_size: int,
            backbone: nn.Module,
            coupling: Coupling,
            kappa: KappaScheduler,
        ) -> None:
            super().__init__()
            self.dict_size = dict_size
            self.backbone = backbone
            self.coupling = coupling
            self.kappa = kappa

        def forward(self, t: float | torch.Tensor, x: Img) -> torch.Tensor:
            """Endpoint class logits ``(b, c, h, w)``."""
            return self.backbone(t, x)

        def step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
            x0_true, x1 = batch

            # flow start x0 w.r.t. coupling (x0: full source, random: uniform noise)
            x0_start, x1 = self.coupling.sample(x0_true, x1)

            # sample t ~ U[0, 1]
            t = torch.rand(len(x1), device=x1.device)

            # sample pt(.|x0, x1)
            dirac_x0 = x2prob(x0_start, self.dict_size)
            dirac_x1 = x2prob(x1, self.dict_size)
            xt = sample_cond_pt(dirac_x0, dirac_x1, t, self.kappa)

            # predict p1|t
            p1t = self(t, xt)
            loss = F.cross_entropy(p1t, x1)
            return loss

    return DiscreteBackbone, DiscreteFM


@app.cell(hide_code=True)
def model_init(DiscreteBackbone, device, dict_size):
    # initial weights, built once; every Train click trains a fresh copy of it
    model_architecture = DiscreteBackbone(dict_size).to(device)
    return (model_architecture,)


@app.cell(hide_code=True)
def path_sample(plt, to_rgb, torch, x0_preview, x1_preview):
    def plot_path_sample(t: float):
        """Plot x₀ | xₜ | x₁ on the linear path (κ_t = t) for the example pair.

        Each pixel shows the target when its uniform draw is below t, the input otherwise.
        """
        x0, x1 = x0_preview[0].cpu(), x1_preview[0].cpu()
        xt = torch.where(torch.rand(x0.shape) < t, x1, x0)
        fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.9))
        panels = [
            (x0, "input x₀ (t = 0)"),
            (xt, f"sample xₜ ~ pₜ(· | x₀, x₁), t = {t:.2f}"),
            (x1, "target x₁ (t = 1)"),
        ]
        for axis, (img, title) in zip(axes, panels, strict=True):
            axis.imshow(to_rgb(img))
            axis.set_title(title, fontsize=9)
            axis.set_xticks(())
            axis.set_yticks(())
            for spine in axis.spines.values():
                spine.set_edgecolor("black")
                spine.set_linewidth(1.2)
        fig.tight_layout()
        return fig

    return (plot_path_sample,)


@app.cell(hide_code=True)
def setup_utilities():
    # Generic plumbing, not specific to discrete flow matching: imports, the U-Net
    # backbone, the toy shape/color dataset with its palette, and diagram helpers.
    import copy
    import math

    import marimo as mo
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import torch
    from diffusers import UNet2DModel
    from einops import rearrange
    from torch import nn
    from torch.nn import functional as F

    MAX_STEPS = 100
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    class FlowUNet(nn.Module):
        """Time-conditioned U-Net predicting endpoint class logits from a flow state.

        Flow time lives in ``[0, 1]``. Diffusers' time embedding is conventionally trained on
        diffusion timesteps, so it is scaled to ``[0, 1000]`` before passing it to the U-Net.
        """

        timestep_scale = 1000.0

        def __init__(self, channels: int, base: int = 32, num_blocks: int = 4) -> None:
            super().__init__()
            self.spatial_multiple = 2 ** (num_blocks - 1)
            self.unet = UNet2DModel(
                sample_size=None,
                in_channels=channels,
                out_channels=channels,
                layers_per_block=1,
                block_out_channels=tuple(base * 2**index for index in range(num_blocks)),
                down_block_types=("DownBlock2D",) * num_blocks,
                up_block_types=("UpBlock2D",) * num_blocks,
                norm_num_groups=8,
                add_attention=False,
            )
            # zero endpoint logits at init = uniform predicted class probabilities
            nn.init.zeros_(self.unet.conv_out.weight)
            nn.init.zeros_(self.unet.conv_out.bias)

        def forward(self, state, time):
            height, width = state.shape[-2:]
            pad_height = (-height) % self.spatial_multiple
            pad_width = (-width) % self.spatial_multiple
            if pad_height or pad_width:
                state = F.pad(state, (0, pad_width, 0, pad_height), mode="reflect")
            prediction = self.unet(state, time * self.timestep_scale, return_dict=False)[0]
            return prediction[..., :height, :width]

    # Dataset: random non-overlapping items that morph along two independent rules.
    #   shape:   square → circle → triangle → square   (index s → (s + 1) % 3)
    #   color:   red → blue → green → red              (index c → (c + 1) % 3)
    # Class index = 1 + color (0 is background): pixels carry only the color; the shape
    # exists only in the geometry of each item's mask.

    SIZE = 32
    RADIUS = 5
    dict_size = 4  # 0 = background, 1 + color for red / blue / green

    SHAPE_NAMES = ["square", "circle", "triangle"]
    COLOR_NAMES = ["red", "blue", "green"]

    # Viz palette: white background, then red / blue / green (one row per class)
    PALETTE = torch.tensor(
        [[1.0, 1.0, 1.0], [0.86, 0.18, 0.18], [0.20, 0.45, 0.92], [0.16, 0.65, 0.30]]
    )

    def _shape_mask(shape: int, cx: float, cy: float, radius: int) -> torch.Tensor:
        axis = torch.arange(SIZE, dtype=torch.float32)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        dx, dy = xx - cx, yy - cy
        if shape == 0:  # square (half-side = radius)
            return (dx.abs() <= radius) & (dy.abs() <= radius)
        if shape == 1:  # circle
            return dx.square() + dy.square() <= float(radius) ** 2
        # triangle: circumradius `radius`, pointing up
        angles = [math.pi / 2 + 2 * math.pi * k / 3 for k in range(3)]
        vertices = [(cx + radius * math.cos(a), cy - radius * math.sin(a)) for a in angles]

        def _edge_sign(o, a, px, py):
            return (a[0] - o[0]) * (py - o[1]) - (a[1] - o[1]) * (px - o[0])

        s0 = _edge_sign(vertices[0], vertices[1], xx, yy)
        s1 = _edge_sign(vertices[1], vertices[2], xx, yy)
        s2 = _edge_sign(vertices[2], vertices[0], xx, yy)
        return ((s0 >= 0) & (s1 >= 0) & (s2 >= 0)) | ((s0 <= 0) & (s1 <= 0) & (s2 <= 0))

    def sample_pair(
        batch_size: int, max_items: int = 5, size: int = SIZE, radius: int = RADIUS
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample (x0, x1): random non-overlapping items, both rules applied in x1.

        Each item keeps its center but its shape is re-drawn morphed
        (square→circle→triangle) and its color attribute cycled (red→blue→green).
        """
        x0 = torch.zeros(batch_size, size, size, dtype=torch.long)
        x1 = torch.zeros_like(x0)
        min_dist = 2 * radius + 2
        for b in range(batch_size):
            n_items = int(torch.randint(2, max_items + 1, (1,)).item())
            centers: list[tuple[float, float]] = []
            for _ in range(n_items):
                for _attempt in range(100):
                    cx = float(torch.randint(radius + 1, size - radius, (1,)).item())
                    cy = float(torch.randint(radius + 1, size - radius, (1,)).item())
                    if all((cx - ox) ** 2 + (cy - oy) ** 2 >= min_dist**2 for ox, oy in centers):
                        centers.append((cx, cy))
                        break
                else:
                    continue  # could not place without overlap; skip this item
                shape = int(torch.randint(0, 3, (1,)).item())
                color = int(torch.randint(0, 3, (1,)).item())
                x0[b][_shape_mask(shape, cx, cy, radius)] = 1 + color
                new_shape, new_color = (shape + 1) % 3, (color + 1) % 3
                x1[b][_shape_mask(new_shape, cx, cy, radius)] = 1 + new_color
        return x0, x1

    def to_rgb(img: torch.Tensor) -> torch.Tensor:
        """(..., H, W) class indices → (..., H, W, 3) palette RGB for display."""
        return PALETTE[img.detach().cpu()]

    def probs_to_rgb(probs: torch.Tensor) -> torch.Tensor:
        """(..., C, H, W) class probabilities → (..., H, W, 3) soft palette RGB."""
        return torch.einsum("...chw,cr->...hwr", probs.detach().cpu().float(), PALETTE)

    def draw_item(
        axis, shape: str, center: tuple[float, float], color, size: float = 0.22
    ) -> None:
        """Draw one dataset item (square half-side / circle radius / triangle circumradius = size)."""
        style = {"facecolor": color, "edgecolor": "black", "linewidth": 1.2}
        cx, cy = center
        if shape == "square":
            patch = mpatches.Rectangle((cx - size, cy - size), 2 * size, 2 * size, **style)
        elif shape == "circle":
            patch = mpatches.Circle(center, size, **style)
        else:
            patch = mpatches.RegularPolygon(center, 3, radius=size * 1.25, **style)
        axis.add_patch(patch)

    def draw_cycle(title: str, labels: list[str], draw_node):
        """Three nodes on a circle, clockwise arrows node k → node k+1 (mod 3)."""
        fig, axis = plt.subplots(figsize=(4.2, 4.2))
        angles = [90, -30, 210]  # clockwise from the top
        radius = 1.0
        centers = [
            (radius * math.cos(math.radians(a)), radius * math.sin(math.radians(a)))
            for a in angles
        ]
        for index, (center, label) in enumerate(zip(centers, labels, strict=True)):
            draw_node(axis, index, center)
            axis.text(center[0], center[1] - 0.42, label, ha="center", va="top", fontsize=10)
        for index in range(3):
            start, end = centers[index], centers[(index + 1) % 3]
            axis.add_patch(
                mpatches.FancyArrowPatch(
                    start,
                    end,
                    connectionstyle="arc3,rad=-0.35",
                    arrowstyle="-|>,head_length=8,head_width=5",
                    shrinkA=30,
                    shrinkB=30,
                    color="0.25",
                    linewidth=1.5,
                )
            )
        axis.set_title(title, fontsize=11)
        axis.set_xlim(-1.6, 1.6)
        axis.set_ylim(-1.35, 1.55)
        axis.set_aspect("equal")
        axis.axis("off")
        fig.tight_layout()
        return fig

    return (
        COLOR_NAMES,
        F,
        FlowUNet,
        MAX_STEPS,
        PALETTE,
        SHAPE_NAMES,
        copy,
        device,
        dict_size,
        draw_cycle,
        draw_item,
        mo,
        nn,
        plt,
        probs_to_rgb,
        rearrange,
        sample_pair,
        to_rgb,
        torch,
    )


@app.cell(hide_code=True)
def randomize_button(mo):
    randomize_button = mo.ui.button(
        label="🎲 Randomize sample",
        value=0,
        on_click=lambda value: value + 1,
    )
    return (randomize_button,)


@app.cell(hide_code=True)
def jump_example_buttons(mo):
    guess_resample_button = mo.ui.button(
        label="🎲 Resample",
        value=0,
        on_click=lambda value: value + 1,
    )
    jump_resample_button = mo.ui.button(
        label="🎲 Resample",
        value=0,
        on_click=lambda value: value + 1,
    )
    return guess_resample_button, jump_resample_button


@app.cell(hide_code=True)
def path_time(mo):
    path_t = mo.ui.slider(
        start=0.0, stop=1.0, step=0.01, value=0.5, label="t", show_value=True, full_width=True
    )
    return (path_t,)


@app.cell(hide_code=True)
def ui_controls(mo):
    coupling_choice = mo.ui.dropdown(
        options={
            "x0 start": "x0",
            "random (uniform noise)": "random",
        },
        value="x0 start",
        label="Coupling",
    )
    kappa_choice = mo.ui.dropdown(
        options={"Linear": "linear", "Cubic scheduler": "cubic"},
        value="Linear",
        label="Time interpolation κ",
    )
    train_steps_choice = mo.ui.slider(start=2, stop=5, value=3, label="10^n training steps")
    return coupling_choice, kappa_choice, train_steps_choice


@app.cell(hide_code=True)
def kappa_params(mo):
    a_choice = mo.ui.slider(
        start=0, stop=3, step=0.25, value=2.0, label="Cubic κ parameter a", show_value=True
    )
    b_choice = mo.ui.slider(
        start=0, stop=3, step=0.25, value=0.5, label="Cubic κ parameter b", show_value=True
    )
    return a_choice, b_choice


@app.cell(hide_code=True)
def train_button(
    a_choice,
    b_choice,
    coupling_choice,
    kappa_choice,
    mo,
    train_steps_choice,
):
    def _snapshot_on_click(_value):
        """Associate the click with the current control values, so training only
        (re)starts on button clicks — moving sliders never triggers a retrain."""
        return {
            "coupling": coupling_choice.value,
            "kappa": kappa_choice.value,
            "a": a_choice.value,
            "b": b_choice.value,
            "steps": train_steps_choice.value,
        }

    train_button = mo.ui.button(
        label="▶ Train model",
        value=None,
        on_click=_snapshot_on_click,
    )
    return (train_button,)


@app.cell(hide_code=True)
def inference_steps(MAX_STEPS, mo):
    inference_step = mo.ui.slider(
        start=1,
        stop=MAX_STEPS,
        step=1,
        value=5,
        label="Number of Euler steps",
        show_value=True,
    )
    return (inference_step,)


@app.cell(hide_code=True)
def trajectory_ui(mo, sample_trajectory):
    sample_iteration = mo.ui.slider(
        start=0,
        stop=len(sample_trajectory) - 1,
        step=1,
        value=len(sample_trajectory) - 1,
        label="Trajectory state",
        show_value=True,
    )
    show_soft_map = mo.ui.switch(
        value=False,
        label="Blend palette by class probability (visible only with 'Show model endpoint prediction p₁')",
    )
    show_model_prediction = mo.ui.switch(
        value=False,
        label="Show model endpoint prediction p₁",
    )
    return sample_iteration, show_model_prediction, show_soft_map


if __name__ == "__main__":
    app.run()
