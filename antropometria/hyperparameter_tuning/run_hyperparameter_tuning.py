import os
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm  # Importa o tqdm

# Suas importações específicas
from antropometria.config import get_logger, BINARY, BINARY_FIELDNAMES, MULTICLASS_FIELDNAMES, CLASSIFIERS, REDUCTIONS, SAMPLINGS, FILTERS, MIN_MAX_NORMALIZATION
from antropometria.utils.load_processed_data import load_processed_data
from antropometria.utils.mappers import map_test_to_dict, map_grid_search_results_to_dict
from antropometria.utils.results import write_header, save_results
from antropometria.hyperparameter_tuning.grid_search import grid_search
from antropometria.error_estimation import run_error_estimation

log = get_logger(__file__)
FIELDNAMES = BINARY_FIELDNAMES if BINARY else MULTICLASS_FIELDNAMES

def process_combination(dataset_name, classes_count, combination):
    reduction, sampling, p_filter, apply_min_max = combination
    output_file = os.path.join('./antropometria/output/GridSearch', f'{dataset_name}_{reduction}_{sampling}_{p_filter}_{apply_min_max}_results.csv')
    write_header(file=output_file, fieldnames=FIELDNAMES)

    try:
        x, y = load_processed_data(dataset_name, apply_min_max, p_filter, reduction, sampling)
        for classifier in CLASSIFIERS:
            accuracy, precision, recall, f1, parameters, best_estimator = grid_search(classifier, x, y)
            current_test = map_test_to_dict(
                dataset_name, classifier.__name__, reduction, p_filter, apply_min_max, sampling
            )
            grid_search_results = map_grid_search_results_to_dict(accuracy, precision, recall, f1)
            error_estimation_results = run_error_estimation(x, y, classes_count, best_estimator, sampling)

            log.info('Saving results!')
            save_results(
                file=output_file,
                fieldnames=FIELDNAMES,
                dataset_shape=x.shape,
                test=current_test,
                grid_search_results=grid_search_results,
                error_estimation_results=error_estimation_results,
                parameters=parameters
            )
    except Exception as error:
        log.error(f'Error processing combination {combination}: {error}')

def run_hyperparameter_tuning(dataset_name: str, classes_count: list[int]):
    log.info(f'Running parallel grid search for {dataset_name}')
    preprocessing_params = list(product(REDUCTIONS, SAMPLINGS, FILTERS, MIN_MAX_NORMALIZATION))
    
    # Inicia o progress bar
    pbar = tqdm(total=len(preprocessing_params), desc="Processing combinations")

    with ProcessPoolExecutor() as executor:
        # Cria uma lista de futures para acompanhar o progresso
        futures = [executor.submit(partial(process_combination, dataset_name, classes_count), param) for param in preprocessing_params]
        
        for future in as_completed(futures):
            # Atualiza o progress bar a cada tarefa concluída
            pbar.update(1)
    
    # Fecha o progress bar após a conclusão de todas as tarefas
    pbar.close()

# Exemplo de como chamar a função
# Substitua "seu_dataset" e "classes_count" pelos valores apropriados
# run_hyperparameter_tuning("seu_dataset", [classes_count])
