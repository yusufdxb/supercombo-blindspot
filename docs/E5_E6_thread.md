1/ Does openpilot's driving model know when it's blind?

Spent the last few weeks doing a distribution-shift teardown of supercombo v0.9.7. Two new results (E5 + E6) just landed. Short version: the collapse is not where you'd guess, and you can see it coming.

2/ Setup recap (E1-E4): we pushed supercombo through a controlled CARLA alpha sweep, slowly walking inputs off-distribution. Around alpha ~= 0.78 the trajectory outputs cliff. Classic "phantom braking" shape, but reproducible and pinned to a number.

3/ Obvious hypothesis: the vision encoder is going blind. OOD pixels saturate something early, features go dead, policy head has nothing to work with. Clean story. Easy to monitor (just watch encoder stats).

It is also wrong.

4/ E5: we instrumented every vision-encoder stage and measured an activity ratio vs the in-distribution baseline along the same sweep.

Result: no encoder stage drops below 0.96 activity. Anywhere. Including past the E4 cliff.

The encoder keeps working. It's downstream.

5/ So the failure mode is not "model can't see." It's "model can see, but the recurrent / policy stack stops using what it sees correctly."

That kills the most intuitive monitor (encoder health) and points the search at a much smaller part of the network.

6/ E6: ok, if the encoder looks fine, what does collapse? We took a rolling spread statistic on the 512-D recurrent feature vector that feeds the policy head. No learned components, no extra training.

Threshold calibrated against real-driving traces to a 1.15% FPR.

7/ Then we re-ran the alpha sweep with the calibrated detector watching.

Detector first fires: alpha = 0.55.
E4 output cliff: alpha ~= 0.78.

The internal-state monitor lights up well before the outputs visibly fail.

8/ That gap matters. A monitor that watches outputs can only react after the model has already emitted a bad trajectory. A monitor on the recurrent feature vector flagged the OOD condition while the outputs still looked plausible.

That's the regime an on-car supervisor actually needs.

9/ Practical takeaways for anyone bolting runtime safety onto a black-box driving model:

- Vision-encoder health checks were not sufficient here. The encoder looked fine the whole way through the cliff.
- A single cheap statistic on the recurrent feature vector gave ~0.23 alpha of headroom before collapse.

10/ Caveats, because this is one model on one sweep:

- We localised the failure to "downstream of the encoder," not yet to a specific named submodule. That's next.
- alpha = 0.55 was demonstrated on the CARLA sweep. Generalisation to other sim engines and real OOD stimuli is still owed.

11/ Two non-obvious gotchas burned a lot of time and are worth flagging for anyone redoing this:

- supercombo's recurrent state MUST roll between frames. Zero-resetting per frame produces a multi-second transient that looks exactly like a phantom brake.
- YUV input is unnormalised. Don't divide by 255.

12/ Repo, code, figures, and the full E5 + E6 writeup:

https://github.com/yusufdxb/supercombo-blindspot

Happy to argue about any of this. Especially interested if anyone has tried similar internal-state monitors on other AV stacks.
