#include <torch/extension.h>
#include <vector>
#include "catCpp.hpp"

// C++ Python interface

torch::Tensor getSpikesCpp(
	torch::Tensor d_u,
	const float theta)
{
	auto d_s = torch::empty_like(d_u);

	// TODO implement for different data types

	unsigned Ns = d_u.size(-1);
	unsigned nNeurons = 1;
	for(int i = 0; i < d_u.ndimension()-1; i++)
		nNeurons *= d_u.size(i);
	getSpikes<float>(d_s.data<float>(), d_u.data<float>(), nNeurons, Ns, theta);

	return d_s;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("getSpikes", &getSpikesCpp, "Get spikes (CPP)");
}
