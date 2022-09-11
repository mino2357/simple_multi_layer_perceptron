# include <Siv3D.hpp> // OpenSiv3D v0.6.5

//
// Visualization of the hidden weight value of a multilayer perceptron (simple multi layer perceptron) on your CPU!
// method: Gradient Method. dx/dt = - dL/dx. L is potential function. In this context, it is an error function.
// todo!: SGD, more layer, Adam method, solving a Burgers equation or KdV eq (this multi layer perceptron possible to solve?) ...
// 

namespace neuron {
	constexpr int net_size = 400; // Resolution
	constexpr int layer = 3; // Number of Perceptron Layers
	constexpr double default_bias = 0.00;
	constexpr double lambda = 0.0001; // L2 regularization
	constexpr int num_train = 200; // Number of training sets
	constexpr int vis_interval = 100; // Interval of visualization
	constexpr int vis_layer_1 = 0;
	constexpr int vis_layer_2 = 1;
	constexpr int vis_layer_3 = layer - 1;
	constexpr double dt = 1.0e-10; // dw/dt = - nabla (Loss-function(w))  (Learning rate)
	constexpr double heat_start = 1.0e-4;
	constexpr double heat_end = 2.0e-4;
	constexpr double vel_ratio = 0.2; // in [0, 1]
	constexpr double heat_ratio = 1.0e-2; // in [0. 1]
	constexpr double target_rmse = 1.0; // finish training value
	constexpr double range_x_min = -0.75; // position of exp(-x^2) function
	constexpr double range_x_max =  0.75;
	constexpr int max_exp_func = 8; // number of exp(-x^2) functions

	// ReLU
	double relu(double x) {
		return std::max(0.0, x);
	}

	class Perceptron {
	public:
		double bias;
		std::vector<std::vector<double>> vec{};

		Perceptron() {
			vec.resize(net_size);
			for (int i = 0; i < net_size; ++i) {
				vec[i].resize(net_size);
			}
			for (int i = 0; i < net_size; ++i) {
				for (int j = 0; j < net_size; ++j) {
					vec[i][j] = 0.0;
				}
			}
			bias = default_bias;
		}
		void resize(int input, int output) {
			vec.resize(input);
			for (int i = 0; i < input; ++i) {
				vec[i].resize(output);
			}
			for (int i = 0; i < net_size; ++i) {
				for (int j = 0; j < net_size; ++j) {
					vec[i][j] = 0.0;
				}
			}
			bias = default_bias;
		}
	};

	class Multi_layer_perceptron {
	public:
		std::vector<Perceptron> pcptrn;
		Multi_layer_perceptron() {
			pcptrn.resize(layer);
			for (int i = 0; i < net_size; ++i) {
				for (int j = 0; j < net_size; ++j) {
					for (int k = 0; k < layer; ++k) {
						pcptrn[k].vec[i][j] = s3d::Random(0.099999, 0.1) / net_size;// 1.0 / net_size; // 1.0 / (1.0 + i + j); // pow(sin(0.04 * (i + j)), 2.0); // 0.5 * std::sin(i * j) + 0.5; // 1.0 / (1.0 + i + j);
					}
				}
			}
		}
		// calc output
		std::vector<double> transmit(std::vector<double> input) {
			auto ret = std::vector<double>{ 0.0 };
			auto layer_tmp = std::vector<double>{ 0.0 };
			layer_tmp.resize(net_size);
			ret.resize(net_size);
			for (int k = 0; k < layer; ++k) {
				for (int i = 0; i < net_size; ++i) {
					for (int j = 0; j < net_size; ++j) {
						layer_tmp[i] += pcptrn[k].vec[i][j] * input[j];
					}
					layer_tmp[i] += pcptrn[k].bias;
				}
				for (int i = 0; i < net_size; ++i) {
					input[i] = neuron::relu(layer_tmp[i]);
				}
			}
			for (int i = 0; i < net_size; ++i) {
				ret[i] = input[i];
			}
			return ret;
		}
	};

	// back-propagation. The following code may be incorrect.(???)
	void renew_weight(const std::vector<double>& input, const std::vector<double>& output, std::vector<Perceptron>& w) {
		std::vector<std::vector<double>> x;
		x.resize(net_size);
		std::vector<double> y;
		y.resize(net_size);
		for (int i = 0; i < net_size; ++i) {
			x[i].resize(net_size);
			y[i] = input[i];
		}

		// forward
		for (int k = 0; k < layer; ++k) {
			for (int i = 0; i < net_size; ++i) {
				for (int j = 0; j < net_size; ++j) {
					x[k][i] += w[k].vec[i][j] * y[i];
				}
				x[k][i] += w[k].bias;
				y[i] = x[k][i];
			}
		}

		// back-propagation
		for (int k = layer - 1; k >= 0; --k) {
			for (int i = 0; i < net_size; ++i) {
				auto diff = -2.0 * (input[i] - output[i]);
				for (int j = 0; j < net_size; ++j) {
					w[k].vec[i][j] = w[k].vec[i][j] - dt * (diff * x[k][j] + lambda * w[k].vec[i][j]);
				}
			}
		}
	}

	// make training data
	// calculation of bellow PDE
	// u_t + u_x = u_xx (+ vel ratio and heat ratio.)
	//
	class Convection_diffusion_eq {
	public:
		std::vector<std::vector<double>> sol_1{};
		std::vector<std::vector<double>> sol_2{};
		Convection_diffusion_eq(int num_sample) {
			Reseed(123456789ull);
			sol_1.resize(num_sample);
			sol_2.resize(num_sample);
			for (int i = 0; i < num_sample; ++i) {
				sol_1[i].resize(net_size);
				sol_2[i].resize(net_size);
				auto x = -1.0;
				auto dx = 2.0 / (net_size - 1);
				auto a = std::vector<double>{};
				auto b = std::vector<double>{};
				auto c = std::vector<double>{};
				auto num_exp_func = max_exp_func; // s3d::Random(1, max_exp_func);
				a.resize(num_exp_func);
				b.resize(num_exp_func);
				c.resize(num_exp_func);
				for (int j = 0; j < num_exp_func; ++j) {
					a[j] = s3d::Random(range_x_min, range_x_max);
					b[j] = 100.0; // s3d::Random(100.0, 100.0);
					c[j] = 1.0; // s3d::Random(1.0, 1.0);
				}
				for (int j = 0; j < sol_1[i].size(); ++j) {
					/*
					　c * exp(- b * (x - a)(x - a)) in [-1, 1]
					*/
					x += dx;
					for (int r = 0; r < num_exp_func; ++r) {
						for (int k = 0; k < net_size; ++k) {
							sol_1[i][j] += c[r] * exp(-b[r] * (x - a[r]) * (x - a[r])) + 1,0e-4;
						}
					}
					sol_2[i][j] = 0.0;
				}
			}
		}
		void make_training_data() {
			double t = 0.0;
			double dx = 1.0 / net_size;
			// stable condition of explicit finite differential method(FDM)
			double dt_heat = 0.4 * dx * dx;
			// CFL condition needs to be vel_ratio in [0,1.0],
			double vel = vel_ratio * (dx / dt_heat);
			// initialization
			std::vector<std::vector<double>> sol_tmp{};
			sol_tmp.resize(sol_1.size());
			for (int i = 0; i < sol_1.size(); ++i) {
				sol_tmp[i].resize(sol_1[i].size());
				for (int j = 0; j < sol_1[i].size(); ++j) {
					sol_tmp[i][j] = 0.0;
				}
			}
			// make initial data
			for (int i = 0; i < sol_1.size(); ++i) {
				for (;;) {
					// explicit FDM
					// vel = const. > 0
					// note: if vel is negative value, discretization of convection term needs using a upwind method.
					for (int j = 1; j < (sol_1[i].size() - 1); ++j) {
						sol_tmp[i][j] = sol_1[i][j]
							+ dt_heat * heat_ratio * (sol_1[i][j + 1] - 2.0 * sol_1[i][j] + sol_1[i][j - 1]) / (dx * dx)
							+ dt_heat * (vel * (sol_1[i][j + 1] - sol_1[i][j])) / dx;
					}
					// periodic boundary condition
					sol_tmp[i][0] = sol_1[i][0]
						+ dt_heat * heat_ratio * (sol_1[i][1] - 2.0 * sol_1[i][0] + sol_1[i][net_size - 1]) / (dx * dx)
						+ dt_heat * (vel * (sol_1[i][1] - sol_1[i][0])) / dx;
					sol_tmp[i][net_size -1] = sol_1[i][net_size - 1]
						+ dt_heat * heat_ratio * (sol_1[i][0] - 2.0 * sol_1[i][net_size - 1] + sol_1[i][net_size - 2]) / (dx * dx)
						+ dt_heat * (vel * (sol_1[i][0] - sol_1[i][net_size - 1])) / dx;
					// renew
					for (int j = 0; j < net_size; ++j) {
						sol_1[i][j] = sol_tmp[i][j];
					}
					// simulation end time
					if (t > heat_start) {
						t = 0.0;
						break;
					}
					t += dt_heat;
				}
			}
			t = 0.0;
			// make exact solution (prediction)
			for (int i = 0; i < sol_1.size(); ++i) {
				for (int j = 0; j < net_size; ++j) {
					sol_2[i][j] = sol_1[i][j];
				}
			}
			for (int i = 0; i < sol_1.size(); ++i) {
				for (;;) {
					// explicit FDM
					for (int j = 1; j < (net_size - 1); ++j) {
						sol_tmp[i][j] = sol_2[i][j]
							+ dt_heat * heat_ratio * (sol_2[i][j + 1] - 2.0 * sol_2[i][j] + sol_2[i][j - 1]) / (dx * dx)
							+ dt_heat * (vel * (sol_2[i][j + 1] - sol_2[i][j])) / dx;
					}
					// periodic boundary condition
					sol_tmp[i][0] = sol_2[i][0]
						+ dt_heat * heat_ratio * (sol_2[i][1] - 2.0 * sol_2[i][0] + sol_2[i][net_size - 1]) / (dx * dx)
						+ dt_heat * (vel * (sol_2[i][1] - sol_2[i][0])) / dx;
					sol_tmp[i][net_size-1] = sol_2[i][net_size - 1]
						+ dt_heat * heat_ratio * (sol_2[i][0] - 2.0 * sol_2[i][net_size-1] + sol_2[i][net_size - 2]) / (dx * dx)
						+ dt_heat * (vel * (sol_2[i][0] - sol_2[i][net_size-1])) / dx;
					for (int j = 0; j < net_size; ++j) {
						sol_2[i][j] = sol_tmp[i][j];
					}
					// simulation end time
					if (t > heat_end) {
						t = 0.0;
						break;
					}
					t += dt_heat;
				}
			}
		}
	};
}

// input x in the range [0,1].
std::tuple<double, double, double> to_rgb(double x) {
	//x = 0.5 * x + 0.5;
	double r = 0.0;
	double g = 0.0;
	double b = 0.0;
	if (0.0 <= x && x < 0.1) {
		r = 0.0;
		g = 4.0 * x;
		b = 10.0 * x;
	}
	else if (0.1 <= x && x < 0.25) {
		r = 0.0;
		g = 4.0 * x;
		b = 1.0;
	}
	else if (0.25 <= x && x < 0.5) {
		r = 0.0;
		g = 1.0;
		b = -4.0 * x + 2.0;
	}
	else if (0.5 <= x && x < 0.75) {
		r = 4.0 * x - 2.0;
		g = 1.0;
		b = 0.0;
	}
	else if (0.75 <= x && x <= 1.001) {
		r = 1.0;
		g = -4.0 * x + 4.0;
		b = 0.0;
	}
	else if (x > 1.001) {
		r = 1.0;
		g = 1.0;
		b = 1.0;
	}
	return std::tuple<double, double, double>(r, g, b);
}

std::vector<double> normalize(const std::vector<double>& vec) {
	auto ret = vec;
	double max = -1.0e9;
	double min = 1.0e9;
	for (int i = 0; i < vec.size(); ++i) {
		max = std::max(max, vec[i]);
		min = std::min(min, vec[i]);
	}
	double dx = max - min;
	for (int i = 0; i < vec.size(); ++i) {
		ret[i] = vec[i] / dx - min / dx;
	}
	return ret;
}

std::vector<std::vector<double>> normalize(const std::vector<std::vector<double>>& vec) {
	auto ret = vec;
	double max = -1.0e9;
	double min = 1.0e9;
	for (int i = 0; i < vec.size(); ++i) {
		for (int j = 0; j < vec[i].size(); ++j) {
			max = std::max(max, vec[i][j]);
			min = std::min(min, vec[i][j]);
		}
	}
	double dx = max - min;
	for (int i = 0; i < vec.size(); ++i) {
		for (int j = 0; j < vec[i].size(); ++j) {
			ret[i][j] = vec[i][j] / dx - min / dx;
		}
	}
	return ret;
}

void Main()
{
	Window::SetTitle(U"mino2357's neuron!");
	// window size
	Window::Resize(1600, 900);
	Window::SetStyle(WindowStyle::Sizable);
	Scene::SetBackground(ColorF{ 0.0, 0.0, 0.0 });
	// size 50 font
	const Font font{ 18 };
	// simple perceptron
	auto s_pt = neuron::Multi_layer_perceptron();
	// input, output, numerical solution
	auto input = std::vector<double>{};
	input.resize(neuron::net_size);
	auto output = std::vector<double>{};
	output.resize(neuron::net_size);
	auto true_output = std::vector<double>{};
	true_output.resize(neuron::net_size);
	//
	int iter = 0;
	auto diff_norm = 0.0;
	// training data
	auto heat = neuron::Convection_diffusion_eq(neuron::num_train);
	heat.make_training_data();
	auto err_min = 1.0e9;
	auto training_done = 0;
	auto num_train = 0;
	auto finish_train = 0;

	while (System::Update())
	{
		ClearPrint();
		//
		double window_x = Scene::Size().x;
		double window_y = Scene::Size().y;
		double dx = window_x / neuron::net_size / 2.0;
		double dy = window_y / neuron::net_size / 2.0;
		//
		if (training_done == 0) {
			for (int i = 0; i < neuron::vis_interval; ++i) {
				int random = Random(neuron::num_train - 1);

				for (int j = 0; j < neuron::net_size; ++j) {
					input[j] = heat.sol_1[random][j];
					true_output[j] = heat.sol_2[random][j];
				}

				// training
				output = s_pt.transmit(input);
				neuron::renew_weight(true_output, output, s_pt.pcptrn);
				num_train++;
			}
			diff_norm = 0.0;
			for (int i = 0; i < neuron::net_size; ++i) {
				diff_norm += 0.5 * pow(true_output[i] - output[i], 2.0);
			}
			diff_norm = sqrt(diff_norm / neuron::net_size);
			err_min = std::min(err_min, diff_norm);
			// training done
			if (diff_norm < neuron::target_rmse && num_train > neuron::num_train) {
				training_done = 1;
				finish_train = num_train;
			}
		}
		

		if (training_done == 1 && iter%100 == 0) {
			// prediction
			auto x = -1.0;
			auto dx_tmp = 2.0 / (neuron::net_size - 1);
			auto a_1 = s3d::Random(-0.5, 0.5);
			auto a_2 = s3d::Random(-0.5, 0.5);
			auto b_1 = s3d::Random(200.0, 200.0);
			auto b_2 = s3d::Random(200.0, 200.0);
			auto c_1 = s3d::Random(1.0, 1.0);
			auto c_2 = s3d::Random(1.0, 1.0);
			for (int i = 0; i < input.size(); ++i) {
				/*
				　c * exp(- b * (x - a)(x - a)) in [-1, 1]
				*/
				x += dx_tmp;
				input[i] = c_1 * exp(-b_1 * (x - a_1) * (x - a_1)) + c_2 * exp(-b_2 * (x - a_2) * (x - a_2));
				true_output[i] = 0.0;
			}
			output = s_pt.transmit(input);
		}

		// visualazation
		s3d::Print << U"{:.14f}"_fmt(diff_norm);
		s3d::Print << U"{:.14f}"_fmt(err_min);
		s3d::Print << U"学習回数：" << num_train;
		s3d::Print << U"中間層数：" << neuron::layer;
		// input
		auto input_vis = normalize(input);
		for (int i = 0; i < neuron::net_size; ++i) {
			auto r = std::get<0>(to_rgb(input_vis[i]));
			auto g = std::get<1>(to_rgb(input_vis[i]));
			auto b = std::get<2>(to_rgb(input_vis[i]));
			RectF{ ((0.0 * neuron::net_size) / 4.0) * dx, i * dy, ((neuron::net_size) / 4.0) * dx, dy }.draw(ColorF{ r, g, b });
		}
		// numerical solution (expected solution)
		auto true_vis = normalize(true_output);
		for (int i = 0; i < neuron::net_size; ++i) {
			auto r = std::get<0>(to_rgb(true_vis[i]));
			auto g = std::get<1>(to_rgb(true_vis[i]));
			auto b = std::get<2>(to_rgb(true_vis[i]));
			RectF{ ((1.0 * neuron::net_size) / 4.0) * dx, i * dy, ((neuron::net_size) / 4.0) * dx, dy }.draw(ColorF{ r, g, b });
		}
		// output solution
		auto output_vis = normalize(output);
		for (int i = 0; i < neuron::net_size; ++i) {
			auto r = std::get<0>(to_rgb(output_vis[i]));
			auto g = std::get<1>(to_rgb(output_vis[i]));
			auto b = std::get<2>(to_rgb(output_vis[i]));
			RectF{ ((2.0 * neuron::net_size) / 4.0) * dx, i * dy, (neuron::net_size / 4.0) * dx, dy }.draw(ColorF{ r, g, b });
		}
		// layer 0
		auto vis_0 = normalize(s_pt.pcptrn[neuron::vis_layer_1].vec);
		for (int i = 0; i < neuron::net_size; ++i) {
			for (int j = 0; j < neuron::net_size; ++j) {
				auto r = std::get<0>(to_rgb(vis_0[i][j]));
				auto g = std::get<1>(to_rgb(vis_0[i][j]));
				auto b = std::get<2>(to_rgb(vis_0[i][j]));
				RectF{ window_x / 2.0 + j * dx, i * dy, dx, dy }.draw(ColorF{ r, g, b });
			}
		}
		// layer 1
		auto vis_1 = normalize(s_pt.pcptrn[neuron::vis_layer_2].vec);
		for (int i = 0; i < neuron::net_size; ++i) {
			for (int j = 0; j < neuron::net_size; ++j) {
				auto r = std::get<0>(to_rgb(vis_1[i][j]));
				auto g = std::get<1>(to_rgb(vis_1[i][j]));
				auto b = std::get<2>(to_rgb(vis_1[i][j]));
				RectF{ j * dx, window_y / 2.0 + i * dy, dx, dy }.draw(ColorF{ r, g, b });
			}
		}
		// layer 2
		auto vis_2 = normalize(s_pt.pcptrn[neuron::vis_layer_3].vec);
		for (int i = 0; i < neuron::net_size; ++i) {
			for (int j = 0; j < neuron::net_size; ++j) {
				auto r = std::get<0>(to_rgb(vis_2[i][j]));
				auto g = std::get<1>(to_rgb(vis_2[i][j]));
				auto b = std::get<2>(to_rgb(vis_2[i][j]));
				RectF{ window_x / 2.0 + j * dx, window_y / 2.0 + i * dy, dx, dy }.draw(ColorF{ r, g, b });
			}
		}
		// thermal graph
		auto width = window_x / neuron::net_size / 2.0;
		auto height = 0.5 * 1.0 / neuron::max_exp_func;
		for (int i = 0; i < neuron::net_size - 1; ++i) {
			Line{ width * i, window_y / 2.0 - height * input[i], width * (i + 1), window_y / 2.0 - height * input[i + 1] }.draw(2, ColorF{ 0, 1, 1 });
		}
		for (int i = 0; i < neuron::net_size-1; ++i) {
			Line{ width * i, window_y / 2.0 - height * true_output[i], width * (i + 1), window_y / 2.0 - height * true_output[i + 1]}.draw(2, ColorF{1, 0, 1});
		}
		for (int i = 0; i < neuron::net_size-1; ++i) {
			Line{ width * i, window_y / 2.0 - height * output[i], width * (i + 1), window_y / 2.0 - height * output[i + 1] }.draw(2, ColorF{ 1, 1, 0 });
		}
		//
		font(U"Input").draw(10 + 0.0 * window_x / 8, 300 * window_y / 720);
		font(U"Numerical solution").draw(10 + window_x / 8.0, 300 * window_y / 720.0);
		font(U"Prediction").draw(10 + 2.0 * window_x / 8.0, 300 * window_y / 720.0);
		//
		font(U"Layer:{}"_fmt(neuron::vis_layer_1)).draw(window_x - 150, 75);
		font(U"Layer:{}"_fmt(neuron::vis_layer_2)).draw(75, window_y - 75);
		font(U"Layer:{}"_fmt(neuron::vis_layer_3)).draw(window_x - 150, window_y - 75);

		iter++;
	}
}
