# HOT-Step engine

The C++17/GGML engine behind HOT-Step: the `ace-server` HTTP server the app runs,
the `ace-lm`, `ace-synth` and `ace-understand` CLIs, the `ace-train` training
toolchain, and the MiniMax-Music3 and YuE2 backends. It is a fork of
[acestep.cpp](https://github.com/ServeurpersoCom/acestep.cpp).

- Engine guide: [docs/dev/engine.md](../docs/dev/engine.md)
- Building: [docs/dev/building.md](../docs/dev/building.md)
- Request JSON, CLI flags and endpoints: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## Acknowledgements

Independent C++ implementation based on
[ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) by ACE Studio and StepFun.
All model weights are theirs, this is just a native backend.

```bibtex
@misc{gong2026acestep,
	title={ACE-Step 1.5: Pushing the Boundaries of Open-Source Music Generation},
	author={Junmin Gong, Yulin Song, Wenxiao Zhao, Sen Wang, Shengyuan Xu, Jing Guo},
	howpublished={\url{https://github.com/ace-step/ACE-Step-1.5}},
	year={2026},
	note={GitHub repository}
}
```

Licence: MIT, see [LICENSE](LICENSE).
