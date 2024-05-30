from datetime import datetime, timedelta, timezone
import logging
import time


def generate_datetime_range(
    start_date: datetime, end_date: datetime, delta: timedelta = timedelta(hours=1)
) -> list[datetime]:
    """Generate a list of datetimes between two datetimes, including the start and end datetimes."""
    current_date = start_date
    date_list = []

    while current_date <= end_date:
        date_list.append(current_date)
        current_date += delta

    return date_list


def init_logger(current_name: str = __name__):
    """Allows for initializing loggers in threads as well."""
    logger = logging.getLogger(current_name)
    if len(logger.handlers) == 0:
        logger.setLevel(logging.INFO)
        sh = logging.StreamHandler()
        sh.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s [%(process)d] - %(message)s")
        )
        logger.addHandler(sh)
    return logger


def enumerate_with_estimate(
    iter,
    desc_str,
    start_ndx=0,
    print_ndx=4,
    backoff=None,
    iter_len=None,
):
    """
    In terms of behavior, `enumerateWithEstimate` is almost identical
    to the standard `enumerate` (the differences are things like how
    our function returns a generator, while `enumerate` returns a
    specialized `<enumerate object at 0x...>`).

    However, the side effects (logging, specifically) are what make the
    function interesting.

    :param iter: `iter` is the iterable that will be passed into
        `enumerate`. Required.

    :param desc_str: This is a human-readable string that describes
        what the loop is doing. The value is arbitrary, but should be
        kept reasonably short. Things like `"epoch 4 training"` or
        `"deleting temp files"` or similar would all make sense.

    :param start_ndx: This parameter defines how many iterations of the
        loop should be skipped before timing actually starts. Skipping
        a few iterations can be useful if there are startup costs like
        caching that are only paid early on, resulting in a skewed
        average when those early iterations dominate the average time
        per iteration.

        NOTE: Using `start_ndx` to skip some iterations makes the time
        spent performing those iterations not be included in the
        displayed duration. Please account for this if you use the
        displayed duration for anything formal.

        This parameter defaults to `0`.

    :param print_ndx: determines which loop interation that the timing
        logging will start on. The intent is that we don't start
        logging until we've given the loop a few iterations to let the
        average time-per-iteration a chance to stablize a bit. We
        require that `print_ndx` not be less than `start_ndx` times
        `backoff`, since `start_ndx` greater than `0` implies that the
        early N iterations are unstable from a timing perspective.

        `print_ndx` defaults to `4`.

    :param backoff: This is used to how many iterations to skip before
        logging again. Frequent logging is less interesting later on,
        so by default we double the gap between logging messages each
        time after the first.

        `backoff` defaults to `2` unless iter_len is > 1000, in which
        case it defaults to `4`.

    :param iter_len: Since we need to know the number of items to
        estimate when the loop will finish, that can be provided by
        passing in a value for `iter_len`. If a value isn't provided,
        then it will be set by using the value of `len(iter)`.

    :return:
    """
    logger = init_logger(__name__)

    if iter_len is None:
        iter_len = len(iter)

    if backoff is None:
        backoff = 2
        while backoff**7 < iter_len:
            backoff *= 2

    assert backoff >= 2
    while print_ndx < start_ndx * backoff:
        print_ndx *= backoff

    logger.warning(
        "{} ----/{}, starting".format(
            desc_str,
            iter_len,
        )
    )
    start_ts = time.time()
    for current_ndx, item in enumerate(iter):
        yield (current_ndx, item)
        if current_ndx == print_ndx:
            # ... <1>
            duration_sec = (
                (time.time() - start_ts)
                / (current_ndx - start_ndx + 1)
                * (iter_len - start_ndx)
            )

            done_dt = datetime.fromtimestamp(start_ts + duration_sec)
            done_td = timedelta(seconds=duration_sec)

            logger.info(
                "{} {:-4}/{}, done at {}, {}".format(
                    desc_str,
                    current_ndx,
                    iter_len,
                    str(done_dt).rsplit(".", 1)[0],
                    str(done_td).rsplit(".", 1)[0],
                )
            )

            print_ndx *= backoff

        if current_ndx + 1 == start_ndx:
            start_ts = time.time()

    logger.warning(
        "{} ----/{}, done at {}".format(
            desc_str,
            iter_len,
            str(datetime.now()).rsplit(".", 1)[0],
        )
    )
