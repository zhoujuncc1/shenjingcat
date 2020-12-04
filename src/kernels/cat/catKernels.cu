#include <torch/extension.h>
#include <vector>
#include "spikeKernels.h"

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_DEVICE(x, y) AT_ASSERTM(x.device().index() == y.device().index(), #x " and " #y " must be in same CUDA device")

// C++ Python interface

torch::Tensor getSpikesCuda(
	torch::Tensor d_u,
	const float theta)
{
	CHECK_INPUT(d_u);

	auto d_s = torch::empty_like(d_u);

	cudaSetDevice(d_u.device().index());

	unsigned Ns = d_u.size(-1);
	unsigned nNeurons = 1;
	for(int i = 0; i < d_u.ndimension()-1; i++)
		nNeurons *= d_u.size(i);
	getSpikes<float>(d_s.data<float>(), d_u.data<float>(), nNeurons, Ns, theta);

	return d_s;
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("getSpikes", &getSpikesCuda, "Get spikes (CUDA)");

}
