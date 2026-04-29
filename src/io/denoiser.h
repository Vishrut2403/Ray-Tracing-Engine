#pragma once

#include "render/framebuffer.h"
#include <OpenImageDenoise/oidn.h>
#include <iostream>
#include <vector>
#include <mutex>

class OIDNDenoiser {
public:
	static void denoise(Framebuffer& fb, bool /*use_cuda*/ = false) {
		const int W = fb.get_width();
		const int H = fb.get_height();
		const int N = W * H;

		std::vector<float> color_in(N * 3);
		std::vector<float> color_out(N * 3, 0.0f);

		for (int j = 0; j < H; ++j) {
			for (int i = 0; i < W; ++i) {
				int idx = j * W + i;
				color c = fb.get(i, j);
				color_in[idx*3+0] = (float)std::max(0.0, c.x());
				color_in[idx*3+1] = (float)std::max(0.0, c.y());
				color_in[idx*3+2] = (float)std::max(0.0, c.z());
			}
		}

		OIDNDevice device = oidnNewDevice(OIDN_DEVICE_TYPE_DEFAULT);
		if (!device) {
			std::cerr << "[OIDN] failed to create device\n";
			return;
		}
		oidnCommitDevice(device);

		const char* err_msg = nullptr;
		if (oidnGetDeviceError(device, &err_msg) != OIDN_ERROR_NONE) {
			std::cerr << "[OIDN] device error: " << err_msg << "\n";
			oidnReleaseDevice(device);
			return;
		}

		OIDNBuffer color_buf = oidnNewBuffer(device, N * 3 * sizeof(float));
		OIDNBuffer output_buf = oidnNewBuffer(device, N * 3 * sizeof(float));

		float* mapped = (float*)oidnGetBufferData(color_buf);
		if (mapped) {
			memcpy(mapped, color_in.data(), N * 3 * sizeof(float));
		} else {
			oidnWriteBuffer(color_buf, 0, N * 3 * sizeof(float), color_in.data());
		}

		OIDNFilter filter = oidnNewFilter(device, "RT");
		oidnSetFilterImage(filter, "color",  color_buf,
						   OIDN_FORMAT_FLOAT3, W, H, 0, 0, 0);
		oidnSetFilterImage(filter, "output", output_buf,
						   OIDN_FORMAT_FLOAT3, W, H, 0, 0, 0);
		oidnSetFilterBool(filter, "hdr", true);
		oidnCommitFilter(filter);

		std::cerr << "[OIDN] denoising " << W << "x" << H << "...\n";
		oidnExecuteFilter(filter);

		if (oidnGetDeviceError(device, &err_msg) != OIDN_ERROR_NONE) {
			std::cerr << "[OIDN] filter error: " << err_msg << "\n";
		} else {
			oidnReadBuffer(output_buf, 0, N * 3 * sizeof(float), color_out.data());

			std::lock_guard<std::mutex> lock(fb.mtx);
			for (int j = 0; j < H; ++j) {
				for (int i = 0; i < W; ++i) {
					int idx = j * W + i;
					fb.set(i, j, color(color_out[idx*3+0],
									   color_out[idx*3+1],
									   color_out[idx*3+2]));
				}
			}
			std::cerr << "[OIDN] done\n";
		}

		oidnReleaseFilter(filter);
		oidnReleaseBuffer(color_buf);
		oidnReleaseBuffer(output_buf);
		oidnReleaseDevice(device);
	}
};