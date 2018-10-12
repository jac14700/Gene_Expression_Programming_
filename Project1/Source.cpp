#include"GEP.cpp"
int main(void)
{
	auto _TrainingData = ReadFile_csv<float>("testfunc_dataset.csv");
	std::size_t _PopulationSize = 100;
	std::size_t _HeadLength = 30;
	std::size_t _ElitismSize = _PopulationSize / 3;
	double _CrossoverRate = 1.0;
	double _MutationRate = 0.2;
	float _ConstantMinimum = -10.0;
	float _ConstantMaximum = 10.0;

	GeneExpressionProgramming _GEP(
		_TrainingData,
		_PopulationSize,
		_HeadLength,
		_ElitismSize,
		_CrossoverRate,
		_MutationRate,
		_ConstantMinimum,
		_ConstantMaximum,
		std::random_device()()
	);

	std::size_t _IteratorCount = 1000;
	std::size_t _PerturbingCount = 100;
	std::size_t _PerturbingSize = _ElitismSize;
	std::vector<float> Fitness(_IteratorCount);

	auto _Begin = std::chrono::high_resolution_clock::now();
	for (decltype(_IteratorCount) _Iterator = 0; _Iterator < _IteratorCount; _Iterator++)
	{
		_GEP.Computing();
		auto _Fitness = _GEP.GetBestChromosome().first;
		std::cout << "Iterator: " << _Iterator << "\t" << "BestMSE: " << _Fitness << std::endl;

		Fitness[_Iterator] = _Fitness;

		if (0 == _Fitness)
		{
			Fitness.resize(_Iterator + 1);
			break;
		}

		if (0 == _Iterator % _PerturbingCount)
		{
			_GEP.Perturbing(_PerturbingSize);
		}
	}

	std::chrono::duration<float> _Difference = std::chrono::high_resolution_clock::now() - _Begin;
	std::cout << std::endl;
	std::cout << "PopulationSize : " << _PopulationSize << std::endl;
	std::cout << "HeadLength : " << _HeadLength << std::endl;
	std::cout << "ElitismSize : " << _ElitismSize << std::endl;
	std::cout << "CrossoverRate : " << _CrossoverRate << std::endl;
	std::cout << "MutationRate : " << _MutationRate << std::endl;
	std::cout << "ConstantMinimum : " << _ConstantMinimum << std::endl;
	std::cout << "ConstantMaximum : " << _ConstantMaximum << std::endl;
	std::cout << "PerturbingCount : " << _PerturbingCount << std::endl;
	std::cout << "PerturbingSize : " << _PerturbingSize << std::endl;
	std::cout << std::endl;
	std::cout << "Time : " << _Difference.count() << "s" << std::endl;
	std::cout << "IteratorCount : " << Fitness.size() << std::endl;
	std::cout << "DataCount : " << _TrainingData.size() << std::endl;
	std::cout << "DataDimension : " << _TrainingData[0].size() << std::endl;
	std::cout << "BestMSE : " << _GEP.GetBestChromosome().first / _TrainingData.size() << std::endl;
	std::cout << std::endl;
	std::cout << "Function : " << _GEP.Decodeing(_GEP.GetBestChromosome().second) << std::endl;
	std::cout << std::endl;

	WriteFile_csv<float>(Fitness, "Fitness.csv");

	return EXIT_SUCCESS;
}