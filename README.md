Наивный байесовский классификатор — простой вероятностный классификатор, основанный на применении Теоремы Байеса со строгими (наивными) предположениями о независимости (формула [1](#orglatexenvironment1)). &#x2014; [Wikipedia](https://ru.wikipedia.org/wiki/Наивный_байесовский_классификатор)

```math
p(C_k | x) = \frac{p(C_k) p(x | C_k)}{p(x)},
```

где $`p(C_k | x)`$ &#x2014; Posterior Probability, $`p(C_k)`$ &#x2014; Class Prior Probability, $`p(x | C_k)`$ &#x2014; Likehood, $`p(x)`$ &#x2014; Predictor Prior Probability.

В данной работе реализован наивный байесовский классификатор на базе Гауссовой функции для данных с нормальным распределением. Модель данного типа используется в случае непрерывных признаков и предполагает, что значения признаков имеют нормальное распределение.

# Задание

Имеются объекты двух классов, характеризующиеся двумя атрибутами `x` и `y`, каждый атрибут представляет собой непрерывную случайную величину, распределённую по нормальному закону. Приведена обучающая выборка двух классов и атрибуты распознаваемого объекта.

Задача: построить классификатор и отнести распознаваемый объект к определённому классу.

```text
X_1 = [8.692, 8.349, 10.644, 9.680, 8.203, 7.681, 6.453, 7.039, 8.568, 9.551]
Y_1 = [11.023, 4.634, 5.380, 7.341, 5.507, 5.459, 5.671, 5.663, 9.348, 7.882]
X_2 = [6.845, 6.564, 7.459, 4.832, 5.148, 7.292, 8.928, 7.738, 5.434, 8.559]
Y_2 = [5.609, 7.641, 6.360, 8.555, 6.509, 5.393, 8.124, 9.005, 8.383, 6.835]
```

```text
X = 8
Y = 8
```

# Описание функционала

Программа состоит из двух модулей: главного ([`main.rs`](./src/main.rs)) и модуля для расчёта Гауссовой функции и Posterior Probability &#x2014; $`P(C_{k} | x)`$ ([`gaussian.rs`](./src/gaussian.rs)).

## Модуль `main.rs`

Данный модуль отвечает за ввод исходных данных (считывание информации о данных для обучения, о классах, об анализируемых данных) и непосредственно за классификацию целевых объектов.

### Формат ввода

После запуска программа попросит ввести количество элементов для обучения и количество атрибутов, по которым будет проводится классификация (листинг [3](#orgsrcblock1)). Производится ввод двух целых чисел через пробел, по окончанию ввод следует нажать `<Enter>`.

```text
Input count of train elements and count of features (e.g. 20 2):
$ 5 2
```

Затем требуется ввести данные для обучения в количестве указанном раннее (листинг [4](#orgsrcblock2)).

```text
Count of elements: 5; count of features: 2.
Input train data (e.g.
1.332 234.2
2.34 3
4 5):
$ 8.692 11.023
$ 8.349 4.634
$ 10.644 5.380
$ 9.680 7.341
$ 6.845 5.609
Train data:
⎡ 8.692 11.023⎤
⎢ 8.349  4.634⎥
⎢10.644   5.38⎥
⎢  9.68  7.341⎥
⎣ 6.845  5.609⎦
```

Также требуется ввести информацию о принадлежности данных к тому или иному классу (листинги [5](#orgsrcblock3), [6](#orgsrcblock4)). Нежелательно, чтобы одному из классов соответствовало менее двух объектов, также недопустимо, чтобы один объект соответствовал нескольким классам. Ввод классов происходит в том же порядке, в каком были введены данные об объектах для обучения.

```text
Input count of classes:
$ 2
```

```text
Input classes data (e.g.
1 0 0
0 0 1
0 1 0):
$ 1 0
$ 1 0
$ 1 0
$ 0 1
$ 0 1
Class data:
⎡1 0⎤
⎢1 0⎥
⎢1 0⎥
⎢0 1⎥
⎣0 1⎦
```

В конце вводится информация для объектов, которые необходимо классифицировать (листинги [7](#orgsrcblock5), [8](#orgsrcblock6)).

```text
Input count of targets for predict:
$ 3
```

```text
Count of elements: 1; count of features: 2
Input target data (e.g.
1.332 234.2 3.4
2.34 3 5.6
4 5 0.0):
$ 1.332 12.3
$ 12 32
$ 1 1
Targets data:
⎡1.332  12.3⎤
⎢   12    32⎥
⎣    1     1⎦
```

В конечном итоге мы получаем информацию о классификации целевых объектов (листинг [9](#orgsrcblock7)).

```text
Class of target #0 is 1
Class of target #1 is 1
Class of target #2 is 2
```

### Тесты

Данный модуль также включает в себя функции самотестирования для выявления ошибок при изменении функционала. Данные тесты запускаются каждый раз, когда происходит слияние рабочих копий в общую основную ветвь разработки при непрерывной интеграции (Continuous Integration) (листинг [10](#orgsrcblock8)).

```text
$ cargo test
   Compiling naive_bayes v0.1.0 (file:///home/.../naive_bayes)
    Finished debug [unoptimized + debuginfo] target(s) in 1.94 secs
     Running target/debug/naive_bayes-2d0129c823898c12

running 1 test
test tests::test_gaussian ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured
```

## Модуль `gaussian.rs`

Представляет из себя структуру данных `Gaussian`, которая содержит в себе информацию о математическом ожидании (`expected`) и дисперсии (`variance`), характерных для определённого класса данных с заданной характеристикой (листинг [11](#orgsrcblock9)). Каждая из полей структуры представляет из себя матрицу (таблица [1](#orgtable1)).

```rust
pub struct Gaussian {
    /// `mu` - expected value.
    expected: Matrix<f64>,
    /// `sigma ^ 2` - variance.
    variance: Matrix<f64>,
}
```

|               | $`feature_{1}`$ | $`feature_{2}`$ | $`feature_{3}`$ | &#x2026; | $`feature_{n}`$ |
|-------------- |---------------- |---------------- |---------------- |--------- |---------------- |
| $`Class_{1}`$ | exp, var        | exp, var        | exp, var        | &#x2026; | exp, var        |
| $`Class_{2}`$ | exp, var        | exp, var        | exp, var        | &#x2026; | exp, var        |
| &#x2026;      | &#x2026;        | &#x2026;        | &#x2026;        | &#x2026; | &#x2026;        |
| $`Class_{k}`$ | exp, var        | exp, var        | exp, var        | &#x2026; | exp, var        |

Статический метод `from_model` (листинг [12](#orgsrcblock10)) на вход принимает количество анализируемых классов и количество атрибутов, возвращает новый объект структуры `Gaussian`.

```rust
/// Create empty parameters and likehoods for data.
pub fn from_model(class_count: usize, features_count: usize) -> Self {
```

Метод `compute_gaussian` (листинг [13](#orgsrcblock11)) на вход принимает матрицу данных для обучения и идентификационный номер класса, для которого требуется вычислить математическое ожидание и дисперсию всех характеристик. Как уже было сказано, данный метод вычисляет необходимые параметры Гауссовой функции для каждого из атрибутов класса и записывает в поля ассоциированной структуры (матрицы `expected` и `variance`).

```rust
/// Compute parameters (expected value, variance) for the Class of data.
pub fn compute_gaussian(&mut self, data: &Matrix<f64>, class_num: usize) {
```

Метод `compute_likehood_and_predict` (листинг [14](#orgsrcblock12)) на вход принимает матрицу объектов (матрица атрибутов), которые необходимо классифицировать, а также массив Class Prior Probability &#x2014; $`P(C_{k})`$. Возвращает матрицу, которая содержит вероятности соотнесения объектов к тому или иному классу.

```rust
/// Compute likehood and Posterior Probability for each class of data.
pub fn compute_likehood_and_predict(&self,
                                    targets: &Matrix<f64>,
                                    class_prior: &[f64])
                                    -> Matrix<f64> {
```

Нормальное распределение, также называемое распределением Гаусса или Гаусса&#x2013;Лапласа —&#x2013; распределение вероятностей, которое в одномерном случае задаётся функцией плотности вероятности, совпадающей с функцией Гаусса (формула [2](#orglatexenvironment2)).

```math
p(x = \upsilon|c) = \frac{1}{\sqrt{2 \pi \sigma_{c}^{2}}} e^{-\frac{(\upsilon - \mu_{c})^{2}}{2 \sigma_{c}^{2}}},
```

где параметр $`\upsilon`$ &#x2013;— математическое ожидание (среднее значение), медиана и мода распределения, а параметр $`\sigma`$ &#x2013;— среднеквадратическое отклонение ($`\sigma^2`$ — дисперсия) распределения.

# Классификация объекта

Для классифицирования заданного объекта приведём данные к требуемому виду (листинг [15](#orgsrcblock13)) и запишем в файл (`variant.inp`).

```text
20 2
8.692 11.023
8.349 4.634
...
7.738 9.005
5.434 8.383
8.559 6.835
2
1 0
1 0
...
0 1
0 1
1
8 8
```

Для классификации передадим содержимое файла с данными программе через пайп (листинг [16](#orgsrcblock14)).

```text
$ cat variant.inp | cargo run --release
    Finished release [optimized] target(s) in 0.0 secs
     Running `target/release/naive_bayes`
...
Targets data:
[8 8]
Class of target #0 is 2
```

# Результаты

В ходе выполнения работы был реализован наивный байесовский классификатор на базе Гауссовой функции для данных с нормальным распределение. Также было выполнено задание по классификации объекта с атрибутами `X = 8; Y = 8`: объект был отнесён ко второму классу.
