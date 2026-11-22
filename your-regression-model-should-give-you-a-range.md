# Your Regression Model Gives You a Number. It Should Give You a Range.

*I built data that knows its own truth, put five different 90 percent intervals on it, and the textbook one covered the truth 66 percent of the time right where the stakes were highest.*

For this exercise I built a dataset that knows the answer before any model touches it. There is one input, call it the size of the account you are forecasting, and one output, the demand. I wrote the true relationship myself, so for any size I know the real average demand and I know exactly how noisy it is. The noise is the part that matters here: it is small for small accounts and large for big ones. The model is sure about the little cases and genuinely uncertain about the big ones, which is what real data looks like.

Then I asked a gradient boosting model for a prediction at one point, an account of size 8. It said 28 units. That number, on its own, is useless for a decision. If you are ordering stock, 28 means something very different when the truth is almost certainly between 26 and 30 than when it could plausibly be 15 or 45. The single number hides the one thing the decision depends on, which is how wrong it might be.

So I asked for a range instead. The standard 90 percent interval at that point ran from about 21 to 36. That looks reasonable. It is also wrong more often than it claims, and the rest of this piece is about how I know that and what to use instead.

![A 90 percent prediction interval drawn over the data. It looks sensible, and it covers the truth about 87 percent of the time.](../outputs/figures/01_first_interval.png)

*A 90 percent interval that looks fine and covers 87 percent. Source: author.*

## The one measurement that judges a range

A prediction interval makes a promise. A 90 percent interval says that if you draw a fresh observation, the true value lands inside the interval 90 percent of the time. That promise is testable. You hold out data the model never saw, you build the intervals, and you count how often the truth falls inside. The fraction you get is the **coverage**, and you compare it to the 90 percent it promised.

This is the whole game. Coverage is to a regression interval what calibration is to a classifier's probabilities, and it gets checked about as often, which is to say almost never. People report an interval, it looks the right size, and nobody ever measures whether it holds. An interval you have not coverage tested is decoration. It is a number with error bars drawn on for comfort.

Everything below is scored this way. I generate data with four different noise shapes, build intervals five ways, and for each one I measure coverage against the known truth. Nominal is 90 percent every time.

## Where the textbook interval quietly fails

The textbook interval is the one every course teaches and every library hands you: take the prediction, add and subtract a fixed multiple of the residual standard deviation, done. One width, applied everywhere. It rests on three assumptions about the errors. They are normal, they are symmetric, and their size is constant across the whole range of predictions.

On data that actually satisfies those assumptions, it works. I generated a version with constant noise and measured coverage at 90 percent, almost exactly. Split the predictions into regions from small to large and coverage stays flat near 90 percent across all of them. When the assumptions hold, the simple interval is correct and cheap, and you should use it.

Now I kept everything the same and let the noise grow with the prediction, which is the realistic case. Overall coverage slipped to 88 percent. Two points below nominal does not sound alarming, and if that were the end of the story I would not have written this. The problem is that the overall number is an average that hides the failure.

![The normal interval on data where the noise grows with x. One constant width is too wide on the left and too narrow on the right.](../outputs/figures/02_heteroscedastic_band.png)

*One constant width cannot fit a fan. Source: author.*

The single width has to be a compromise. It is far too wide for the small, quiet accounts and far too narrow for the big, volatile ones. To see it, I measured coverage separately in each region of the prediction range instead of pooling everything together.

![Coverage measured by region. It is near 100 percent in the calm region and about 66 percent in the volatile region.](../outputs/figures/02_heteroscedastic_conditional.png)

*Coverage by region. The flat line is the promise; the curve is reality. Source: author.*

In the calm region the interval covers close to 100 percent of the time. It is so wide there that it almost never misses, which means it is wasting width on cases you were never worried about. In the volatile region, where a bad estimate actually costs you, coverage falls to about 66 percent. The interval is advertised at 90 and delivers two out of three exactly where you needed it most. That is worse than useless, because it reports false confidence at the precise moment the decision is risky.

Skewed errors break the same interval a different way. A symmetric band centered on the prediction cannot match a lopsided error distribution, so it leaves slack on one side and comes up short on the other. The assumptions are not pedantic fine print. They are load bearing, and real data removes them.

## Letting the width breathe

The fix for a width that should change is to stop predicting the mean and predict the edges directly. Instead of one model aimed at the center, train one aimed at the 5th percentile and another at the 95th. The gap between them is your interval, and because each edge is its own fit, the gap can be narrow where the data is tight and wide where it is not.

You get this by swapping the loss function. Ordinary regression minimizes squared error, which chases the mean. The pinball loss, also called the quantile loss, penalizes misses on one side more than the other, and the balance point you choose is the quantile you recover. I wrote a small quantile regressor from scratch on the pinball loss, and I also used the version that is already sitting in most gradient boosting libraries, where you just pass the quantile as an argument and fit two models.

Both of them recover the shape the fixed width could not.

![Coverage by region for the normal interval against the two quantile methods. The quantile curves sit near the 90 percent line across the whole range.](../outputs/figures/03_conditional_comparison.png)

*Quantile methods hold their coverage across regions where the fixed width collapses. Source: author.*

The quantile interval is tight for small accounts and wide for large ones, which is what the data asked for all along. Coverage by region stops sagging in the volatile zone. This is a real improvement and for many problems it is enough.

There are two honest costs. First, you have to retrain. These methods are a property of the model, so you cannot bolt them onto a model you already shipped. Second, and this is the one that bothers me, they come with no promise. The quantile models aim at 90 percent coverage, and in my runs they landed around 94, a little loose rather than a little tight. Better than the textbook interval, but still a number you have to check rather than a guarantee you can rely on.

## A range with a promise

There is a method that gives you the promise, wraps around any model you already have without retraining it, and assumes almost nothing about the data. It is called conformal prediction, it has been in the statistics literature for two decades, and most working data scientists have never used it.

The split conformal version is almost embarrassingly simple. Take your trained model. Hold out a chunk of data it never saw, the calibration set. Run the model on it and record the absolute errors. The 90 percent interval is just the prediction plus and minus the right quantile of those recorded errors.

```python
# model is already trained; x_calib, y_calib were held out
residuals = np.abs(y_calib - model.predict(x_calib))
k = int(np.ceil((len(residuals) + 1) * 0.90))
q = np.sort(residuals)[k - 1]            # the 90 percent error quantile
lower = model.predict(x_new) - q
upper = model.predict(x_new) + q
```

That is the whole idea. The step that looks like a detail, taking the empirical quantile of the actual errors instead of trusting a formula, is what does the work. You never assume the errors are normal or symmetric. You measure them. Under one mild condition, that the calibration data and the future data are drawn the same way, this interval covers at the rate you asked for. Not approximately, not when the assumptions hold, but as a mathematical property of the construction.

I measured it on all four noise shapes. Constant noise, growing noise, skewed, heavy tailed: coverage came back at 91 percent every time. Then I did the thing conformal is quietly good at. I trained a plain gradient booster, wrapped split conformal around it using only a calibration set, changed nothing about the model, and it covered at 90 percent. No retraining, no assumptions, a few lines of code.

Plain conformal has one weakness, the same one the textbook interval had: it produces a single constant width. So coverage is right on average but not in every region. The repair is to compute that error quantile separately in each region of the prediction instead of once globally. This is adaptive, or Mondrian, conformal, and it keeps the guarantee while letting the width vary.

![Adaptive conformal on the growing-noise data. The band is narrow for small predictions and wide for large ones, and coverage holds across regions.](../outputs/figures/04_adaptive_band.png)

*Adaptive conformal: a width that changes by region, with the coverage promise intact. Source: author.*

## The scoreboard

I put every method against every noise shape and scored them on coverage and on width, because at equal coverage the tighter interval is the better one.

![Coverage for every method across every error structure. Conformal and the quantile methods stay green across the board; the normal interval dips; bootstrap is red everywhere.](../outputs/figures/05_coverage_heatmap.png)

*Coverage by method and error structure, nominal 90 percent. Source: author.*

A few things stand out. The normal interval is fine on constant noise and drifts off everywhere else. The quantile methods and both conformal methods cover broadly. Conformal is the only one carrying a guarantee rather than a hope, and adaptive conformal gives the best mix of coverage and tightness: on the growing-noise data it matched the coverage of plain conformal with a noticeably narrower average width.

One method is red across the entire board, and it is worth saying why. The bootstrap interval, where you resample the data, refit many times, and take the spread of the predictions, covered about 4 percent of the time. That is not a bug. Plain bootstrap measures how much the fitted line wiggles, which is uncertainty about the average, not uncertainty about the next observation. With thousands of rows the line barely wiggles, so the interval is almost a point. It answers a real question, just not the one a prediction interval asks. If you reach for the bootstrap to get a predictive range, measure its coverage before you trust it.

![Coverage against width for every method and structure. The goal is to sit on the 90 percent line as far down as possible.](../outputs/figures/05_coverage_vs_width.png)

*Coverage against width. Far right and low is the place to be. Source: author.*

I also wanted to know whether the headline result was a fluke of one random seed, so I regenerated the growing-noise data fifteen times and remeasured. The normal interval averaged 87 percent and dropped as low as 85. Split conformal averaged 90. Adaptive conformal averaged 91 and never fell below 90. The ordering did not move.

## When the plain interval is good enough

I am not telling you to throw out the textbook interval. On clean data with roughly constant, roughly symmetric, roughly normal errors, it is correct, it is one line, and anything fancier is wasted effort. I measured it at 90 percent on exactly that kind of data. The failure is not that the formula is bad. It is that people apply it without ever checking whether their data is the kind it was built for, and real data usually is not.

The decision is not hard once you look. If your errors are constant and well behaved, use the simple interval. If the error size changes across the range, use quantile regression, gradient boosting quantiles, or adaptive conformal, because all three let the width move. If you have a model you cannot or will not retrain, wrap it in split conformal, which needs nothing but a calibration set. If the decision is high stakes and you need an actual guarantee rather than an interval that usually behaves, use conformal, because it is the only one of these that gives you one.

And the rule that sits above all of them: measure coverage on held-out data before you ship. It is a count. It takes a few lines. Every claim in this piece is one of those counts, nothing more. The reason the textbook interval gets away with covering two times out of three is that the count almost never gets run.

The full code, the synthetic data generators, all five methods built from scratch, and the notebooks that produced every figure here are in the repository: [prediction_intervals_arena](https://github.com/Dima806/prediction_intervals_arena). It runs end to end on a two core machine with no GPU.

If you found this useful, the same coverage-against-known-truth idea runs through a few related write ups: [calibration_arena](https://github.com/Dima806/calibration_arena) does it for classifier probabilities, [bootstrap_101](https://github.com/Dima806/bootstrap_101) takes the resampling machinery apart, and [forecasting_arena](https://github.com/Dima806/forecasting_arena) puts intervals on the time axis, where the exchangeability condition conformal relies on needs more care.
