nvml_initialized = False


def get_reason(val):
    throttle = {
        1: 'gpu idle',
        2: 'applications clocks setting',
        4: 'sw power cap',
        8: 'hw slowdown',
        16: 'sync boost',
        32: 'sw thermal slowdown',
        64: 'hw thermal slowdown',
        128: 'hw power brake slowdown',
        256: 'display clock setting',
    }
    reason = ', '.join([throttle[i] for i in throttle if i & val])
    return reason if len(reason) > 0 else 'ok'

def get_nvml():
    global nvml_initialized # pylint: disable=global-statement
    try:
        if not nvml_initialized:
            from installer import install, log
            install('pynvml', quiet=True)
            import pynvml # pylint: disable=redefined-outer-name
            pynvml.nvmlInit()
            log.debug('NVML initialized')
            nvml_initialized = True
        devices = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            dev = pynvml.nvmlDeviceGetHandleByIndex(i)
            device = {
                'name': pynvml.nvmlDeviceGetName(dev),
                'version': {
                    'cuda': pynvml.nvmlSystemGetCudaDriverVersion(),
                    'driver': pynvml.nvmlSystemGetDriverVersion(),
                    'vbios': pynvml.nvmlDeviceGetVbiosVersion(dev),
                    'rom': pynvml.nvmlDeviceGetInforomImageVersion(dev),
                    'capabilities': pynvml.nvmlDeviceGetCudaComputeCapability(dev),
                },
                'pci': {
                    'link': pynvml.nvmlDeviceGetCurrPcieLinkGeneration(dev),
                    'width': pynvml.nvmlDeviceGetCurrPcieLinkWidth(dev),
                    'busid': pynvml.nvmlDeviceGetPciInfo(dev).busId,
                    'deviceid': pynvml.nvmlDeviceGetPciInfo(dev).pciDeviceId,
                },
                'memory': {
                    'total': round(pynvml.nvmlDeviceGetMemoryInfo(dev).total/1024/1024, 2),
                    'free': round(pynvml.nvmlDeviceGetMemoryInfo(dev).free/1024/1024,2),
                    'used': round(pynvml.nvmlDeviceGetMemoryInfo(dev).used/1024/1024,2),
                },
                'clock': { # gpu, sm, memory
                    'gpu': [pynvml.nvmlDeviceGetClockInfo(dev, 0), pynvml.nvmlDeviceGetMaxClockInfo(dev, 0)],
                    'sm': [pynvml.nvmlDeviceGetClockInfo(dev, 1), pynvml.nvmlDeviceGetMaxClockInfo(dev, 1)],
                    'memory': [pynvml.nvmlDeviceGetClockInfo(dev, 2), pynvml.nvmlDeviceGetMaxClockInfo(dev, 2)],
                },
                'load': {
                    'gpu': round(pynvml.nvmlDeviceGetUtilizationRates(dev).gpu),
                    'memory': round(pynvml.nvmlDeviceGetUtilizationRates(dev).memory),
                    'temp': pynvml.nvmlDeviceGetTemperature(dev, 0),
                    'fan': pynvml.nvmlDeviceGetFanSpeed(dev),
                },
                'power': [round(pynvml.nvmlDeviceGetPowerUsage(dev)/1000, 2), round(pynvml.nvmlDeviceGetEnforcedPowerLimit(dev)/1000, 2)],
                'state': get_reason(pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(dev)),
            }
            devices.append(device)
        # log.debug(f'nmvl: {devices}')
        return devices
    except Exception as e:
        log.error(f'NVML: {e}')
        return []
