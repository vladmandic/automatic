import os
import sys
import time
import datetime
from modules.errors import log


class State:
    skipped = False
    interrupted = False
    paused = False
    job = ""
    job_no = 0
    job_count = 0
    frame_count = 0
    total_jobs = 0
    job_timestamp = '0'
    sampling_step = 0
    sampling_steps = 0
    current_latent = None
    current_image = None
    current_image_sampling_step = 0
    id_live_preview = 0
    textinfo = None
    api = False
    time_start = None
    need_restart = False
    server_start = time.time()
    oom = False
    debug_output = os.environ.get('SD_STATE_DEBUG', None)

    def skip(self):
        log.debug('Requested skip')
        self.skipped = True

    def interrupt(self):
        log.debug('Requested interrupt')
        self.interrupted = True

    def pause(self):
        self.paused = not self.paused
        log.debug(f'Requested {"pause" if self.paused else "continue"}')

    def nextjob(self):
        import modules.devices
        self.do_set_current_image()
        self.job_no += 1
        self.sampling_step = 0
        self.current_image_sampling_step = 0
        modules.devices.torch_gc()

    def dict(self):
        obj = {
            "skipped": self.skipped,
            "interrupted": self.interrupted,
            "job": self.job,
            "job_count": self.job_count,
            "job_timestamp": self.job_timestamp,
            "job_no": self.job_no,
            "sampling_step": self.sampling_step,
            "sampling_steps": self.sampling_steps,
        }
        return obj

    def status(self):
        from modules import progress
        from modules.api import models
        res = models.ResStatus(
            task=self.job,
            id=progress.current_task or '',
            job=max(self.job_no, 0),
            jobs=max(self.frame_count, self.job_count, self.job_no),
            total=self.total_jobs,
            timestamp=self.job_timestamp if self.job != '' else None,
            step=self.sampling_step,
            steps=self.sampling_steps,
            queued=len(progress.pending_tasks),
            status='unknown',
            uptime = round(time.time() - self.server_start)
        )
        res.step = res.steps * res.job + res.step
        res.steps = res.steps * res.jobs
        res.progress = round(min(1, abs(res.step / res.steps) if res.steps > 0 else 0), 2)
        res.elapsed = round(time.time() - self.time_start, 2) if self.time_start is not None else None
        predicted = round(res.elapsed / res.progress, 2) if res.progress > 0 and res.elapsed is not None else None
        res.eta = round(predicted - res.elapsed, 2) if predicted is not None else None
        if self.paused:
            res.status = 'paused'
        elif self.interrupted:
            res.status = 'interrupted'
        elif self.skipped:
            res.status = 'skipped'
        else:
            res.status = 'running' if self.job != '' else 'idle'
        return res

    def begin(self, title="", api=None):
        import modules.devices
        self.total_jobs += 1
        self.current_image = None
        self.current_image_sampling_step = 0
        self.current_latent = None
        self.id_live_preview = 0
        self.interrupted = False
        self.job = title
        self.job_count = -1
        self.frame_count = -1
        self.job_no = 0
        self.job_timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.paused = False
        self.sampling_step = 0
        self.skipped = False
        self.textinfo = None
        self.api = api if api is not None else self.api
        self.time_start = time.time()
        if self.debug_output:
            log.debug(f'State begin: {self.job}')
        modules.devices.torch_gc()

    def end(self, api=None):
        import modules.devices
        if self.time_start is None: # someone called end before being
            fn = f'{sys._getframe(2).f_code.co_name}:{sys._getframe(1).f_code.co_name}' # pylint: disable=protected-access
            log.debug(f'Access state.end: {fn}') # pylint: disable=protected-access
            self.time_start = time.time()
        if self.debug_output:
            log.debug(f'State end: {self.job} time={time.time() - self.time_start:.2f}')
        self.job = ""
        self.job_count = 0
        self.job_no = 0
        self.frame_count = 0
        self.paused = False
        self.interrupted = False
        self.skipped = False
        self.api = api if api is not None else self.api
        modules.devices.torch_gc()

    def set_current_image(self):
        from modules.shared import opts, cmd_opts
        """sets self.current_image from self.current_latent if enough sampling steps have been made after the last call to this"""
        if cmd_opts.lowvram or self.api:
            return
        if abs(self.sampling_step - self.current_image_sampling_step) >= opts.show_progress_every_n_steps and opts.live_previews_enable and opts.show_progress_every_n_steps > 0:
            self.do_set_current_image()

    def do_set_current_image(self):
        if self.current_latent is None:
            return
        from modules.shared import opts
        import modules.sd_samplers # pylint: disable=W0621
        try:
            image = modules.sd_samplers.samples_to_image_grid(self.current_latent) if opts.show_progress_grid else modules.sd_samplers.sample_to_image(self.current_latent)
            self.assign_current_image(image)
            self.current_image_sampling_step = self.sampling_step
        except Exception:
            # log.error(f'Error setting current image: step={self.sampling_step} {e}')
            pass

    def assign_current_image(self, image):
        self.current_image = image
        self.id_live_preview += 1
