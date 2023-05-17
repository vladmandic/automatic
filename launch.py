from dockersetup import start_server, get_memory_stats
import time
import installer

if __name__ == "__main__":
    instance = start_server(immediate=True, server=None)
    while True:
        try:
            alive = instance.thread.is_alive()
        except:
            alive = False
        if round(time.time()) % 30 == 0:
            installer.log.debug(f'Server alive: {alive} Memory {get_memory_stats()}')
        if not alive:
            if instance.wants_restart:
                installer.log.info('Server restarting...')
                instance = start_server(immediate=False, server=instance)
            else:
                installer.log.info('Exiting...')
                break
        time.sleep(1)
