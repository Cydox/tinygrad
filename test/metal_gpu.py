import Metal, Cocoa, libdispatch # type: ignore


device = Metal.MTLCreateSystemDefaultDevice()
print(device.name())
