//-----------------------------------------------------------------------------
// Copyright 2012 Masanori Morise
// Author: mmorise [at] yamanashi.ac.jp (Masanori Morise)
// Last update: 2018/01/21
//
// F0 estimation based on Harvest.
//-----------------------------------------------------------------------------
#include "world/harvest.h"

#include <math.h>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "world/common.h"
#include "world/constantnumbers.h"
#include "world/fft.h"
#include "world/matlabfunctions.h"

#define NUM_HARMONICS 6

//-----------------------------------------------------------------------------
// struct for RawEventByHarvest()
// "negative" means "zero-crossing point going from positive to negative"
// "positive" means "zero-crossing point going from negative to positive"
//-----------------------------------------------------------------------------
typedef struct {
  double *negative_interval_locations;
  double *negative_intervals;
  int number_of_negatives;
  double *positive_interval_locations;
  double *positive_intervals;
  int number_of_positives;
  double *peak_interval_locations;
  double *peak_intervals;
  int number_of_peaks;
  double *dip_interval_locations;
  double *dip_intervals;
  int number_of_dips;
  int *negative_going_points;
  int *edges;
  double *fine_edges;
} ZeroCrossings;

namespace {
struct GetRawF0Temp {
  ZeroCrossings zero_crossings;
  double *filtered_signal;
  double *band_pass_filter;
  fft_complex *band_pass_filter_spectrum;

  double *interp1_s;
  int *interp1_k;
  world_t **interpolated_f0_set;  
};

struct RefineF0Temp {
  fft_complex *main_spectrum;
  fft_complex *diff_spectrum;
  int *base_index;
  double *main_window;
  double *diff_window;
  double *power_spectrum;
  double *numerator_i;
  double *base_time;
  int *safe_index;
  ForwardRealFFT *forward_real_fft_list;
};

//-----------------------------------------------------------------------------
// Since the waveform of beginning and ending after decimate include noise,
// the input waveform is extended. This is the processing for the
// compatibility with MATLAB version.
//-----------------------------------------------------------------------------
static void GetWaveformAndSpectrumSub(const double *x, int x_length,
    int y_length, double actual_fs, int decimation_ratio, double *y) {
  if (decimation_ratio == 1) {
    for (int i = 0; i < x_length; ++i) y[i] = x[i];
    return;
  }

  int lag =
    static_cast<int>(ceil(140.0 / decimation_ratio) * decimation_ratio);
  int new_x_length = x_length + lag * 2;
  double *new_y = new double[new_x_length];
  for (int i = 0; i < new_x_length; ++i) new_y[i] = 0.0;
  double *new_x = new double[new_x_length];
  for (int i = 0; i < lag; ++i) new_x[i] = x[0];
  for (int i = lag; i < lag + x_length; ++i) new_x[i] = x[i - lag];
  for (int i = lag + x_length; i < new_x_length; ++i)
    new_x[i] = x[x_length - 1];

  decimate(new_x, new_x_length, decimation_ratio, new_y);
  for (int i = 0; i < y_length; ++i) y[i] = new_y[lag / decimation_ratio + i];

  delete[] new_x;
  delete[] new_y;
}

//-----------------------------------------------------------------------------
// GetWaveformAndSpectrum() calculates the downsampled signal and its spectrum
//-----------------------------------------------------------------------------
static void GetWaveformAndSpectrum(const double *x, int x_length,
    int y_length, double actual_fs, int fft_size, int decimation_ratio,
    double *y, fft_complex *y_spectrum) {
  // Initialization
  for (int i = 0; i < fft_size; ++i) y[i] = 0.0;

  // Processing for the compatibility with MATLAB version
  GetWaveformAndSpectrumSub(x, x_length, y_length, actual_fs,
      decimation_ratio, y);

  // Removal of the DC component (y = y - mean value of y)
  double mean_y = 0.0;
  for (int i = 0; i < y_length; ++i) mean_y += y[i];
  mean_y /= y_length;
  for (int i = 0; i < y_length; ++i) y[i] -= mean_y;
  for (int i = y_length; i < fft_size; ++i) y[i] = 0.0;

  fft_plan forwardFFT =
    fft_plan_dft_r2c_1d(fft_size, y, y_spectrum, FFT_ESTIMATE);
  fft_execute(forwardFFT);

  fft_destroy_plan(forwardFFT);
}

//-----------------------------------------------------------------------------
// GetFilteredSignal() calculates the signal that is the convolution of the
// input signal and band-pass filter.
//-----------------------------------------------------------------------------
static void GetFilteredSignal(double boundary_f0, int fft_size, double fs,
    const fft_complex *y_spectrum, int y_length, double *filtered_signal,
    double *band_pass_filter, fft_complex *band_pass_filter_spectrum) {
  int filter_length_half = lround(fs / boundary_f0 * 2.0);
  NuttallWindow(filter_length_half * 2 + 1, band_pass_filter);
  for (int i = -filter_length_half; i <= filter_length_half; ++i)
    band_pass_filter[i + filter_length_half] *=
      cos(2 * world::kPi * boundary_f0 * i / fs);
  for (int i = filter_length_half * 2 + 1; i < fft_size; ++i)
    band_pass_filter[i] = 0.0;

  fft_plan forwardFFT = fft_plan_dft_r2c_1d(fft_size, band_pass_filter,
    band_pass_filter_spectrum, FFT_ESTIMATE);
  fft_execute(forwardFFT);

  // Convolution
  double tmp = y_spectrum[0][0] * band_pass_filter_spectrum[0][0] -
    y_spectrum[0][1] * band_pass_filter_spectrum[0][1];
  band_pass_filter_spectrum[0][1] =
    y_spectrum[0][0] * band_pass_filter_spectrum[0][1] +
    y_spectrum[0][1] * band_pass_filter_spectrum[0][0];
  band_pass_filter_spectrum[0][0] = tmp;
  for (int i = 1; i <= fft_size / 2; ++i) {
    tmp = y_spectrum[i][0] * band_pass_filter_spectrum[i][0] -
      y_spectrum[i][1] * band_pass_filter_spectrum[i][1];
    band_pass_filter_spectrum[i][1] =
      y_spectrum[i][0] * band_pass_filter_spectrum[i][1] +
      y_spectrum[i][1] * band_pass_filter_spectrum[i][0];
    band_pass_filter_spectrum[i][0] = tmp;
    band_pass_filter_spectrum[fft_size - i - 1][0] =
      band_pass_filter_spectrum[i][0];
    band_pass_filter_spectrum[fft_size - i - 1][1] =
      band_pass_filter_spectrum[i][1];
  }

  fft_plan inverseFFT = fft_plan_dft_c2r_1d(fft_size,
    band_pass_filter_spectrum, filtered_signal, FFT_ESTIMATE);
  fft_execute(inverseFFT);

  // Compensation of the delay.
  int index_bias = filter_length_half + 1;
  for (int i = 0; i < y_length; ++i)
    filtered_signal[i] = filtered_signal[i + index_bias];

  fft_destroy_plan(inverseFFT);
  fft_destroy_plan(forwardFFT);
}

//-----------------------------------------------------------------------------
// CheckEvent() returns 1, provided that the input value is over 1.
// This function is for RawEventByDio().
//-----------------------------------------------------------------------------
static inline int CheckEvent(int x) {
  return x > 0 ? 1 : 0;
}

//-----------------------------------------------------------------------------
// ZeroCrossingEngine() calculates the zero crossing points from positive to
// negative.
//-----------------------------------------------------------------------------
static int ZeroCrossingEngine(const double *filtered_signal, int y_length,
    double fs, double *interval_locations, double *intervals,
    int *negative_going_points, int *edges, double *fine_edges) {

  for (int i = 0; i < y_length - 1; ++i)
    negative_going_points[i] =
      0.0 < filtered_signal[i] && filtered_signal[i + 1] <= 0.0 ? i + 1 : 0;
  negative_going_points[y_length - 1] = 0;

  int count = 0;
  for (int i = 0; i < y_length; ++i)
    if (negative_going_points[i] > 0)
      edges[count++] = negative_going_points[i];

  if (count < 2) {
    return 0;
  }

  for (int i = 0; i < count; ++i)
    fine_edges[i] = edges[i] - filtered_signal[edges[i] - 1] /
      (filtered_signal[edges[i]] - filtered_signal[edges[i] - 1]);

  for (int i = 0; i < count - 1; ++i) {
    intervals[i] = fs / (fine_edges[i + 1] - fine_edges[i]);
    interval_locations[i] = (fine_edges[i] + fine_edges[i + 1]) / 2.0 / fs;
  }

  return count - 1;
}

//-----------------------------------------------------------------------------
// InitializeZeroCrossings() allocates the memory of array in the struct
//-----------------------------------------------------------------------------
static void InitializeZeroCrossings(ZeroCrossings *zero_crossings, int y_length) {
    zero_crossings->negative_interval_locations = new double[y_length];
    zero_crossings->positive_interval_locations = new double[y_length];
    zero_crossings->peak_interval_locations = new double[y_length];
    zero_crossings->dip_interval_locations = new double[y_length];
    zero_crossings->negative_intervals = new double[y_length];
    zero_crossings->positive_intervals = new double[y_length];
    zero_crossings->peak_intervals = new double[y_length];
    zero_crossings->dip_intervals = new double[y_length];
    zero_crossings->negative_going_points = new int[y_length];
    zero_crossings->edges = new int[y_length];
    zero_crossings->fine_edges = new double[y_length];
}


//-----------------------------------------------------------------------------
// GetFourZeroCrossingIntervals() calculates four zero-crossing intervals.
// (1) Zero-crossing going from negative to positive.
// (2) Zero-crossing going from positive to negative.
// (3) Peak, and (4) dip. (3) and (4) are calculated from the zero-crossings of
// the differential of waveform.
//-----------------------------------------------------------------------------
static void GetFourZeroCrossingIntervals(GetRawF0Temp &tmp, int y_length,
    double actual_fs, ZeroCrossings *zero_crossings) {

  zero_crossings->number_of_negatives = ZeroCrossingEngine(tmp.filtered_signal,
    y_length, actual_fs, zero_crossings->negative_interval_locations,
    zero_crossings->negative_intervals,
    zero_crossings->negative_going_points,
    zero_crossings->edges, zero_crossings->fine_edges);

  for (int i = 0; i < y_length; ++i) tmp.filtered_signal[i] = -tmp.filtered_signal[i];
  zero_crossings->number_of_positives = ZeroCrossingEngine(tmp.filtered_signal,
    y_length, actual_fs, zero_crossings->positive_interval_locations,
    zero_crossings->positive_intervals,
    zero_crossings->negative_going_points,
    zero_crossings->edges, zero_crossings->fine_edges);

  for (int i = 0; i < y_length - 1; ++i) tmp.filtered_signal[i] =
    tmp.filtered_signal[i] - tmp.filtered_signal[i + 1];
  zero_crossings->number_of_peaks = ZeroCrossingEngine(tmp.filtered_signal,
    y_length - 1, actual_fs, zero_crossings->peak_interval_locations,
    zero_crossings->peak_intervals,
    zero_crossings->negative_going_points,
    zero_crossings->edges, zero_crossings->fine_edges);

  for (int i = 0; i < y_length - 1; ++i)
    tmp.filtered_signal[i] = -tmp.filtered_signal[i];
  zero_crossings->number_of_dips = ZeroCrossingEngine(tmp.filtered_signal,
    y_length - 1, actual_fs, zero_crossings->dip_interval_locations,
    zero_crossings->dip_intervals,
    zero_crossings->negative_going_points,
    zero_crossings->edges, zero_crossings->fine_edges);
}

//-----------------------------------------------------------------------------
// GetF0CandidateContour() calculates the F0 candidate contour in 1-ch signal.
//-----------------------------------------------------------------------------
static void GetF0CandidateContour(const ZeroCrossings *zero_crossings,
    double boundary_f0, const HarvestOption *option,
    const double *temporal_positions, int f0_length, double *f0_candidate,
    double *const *interpolated_f0_set, double *interp1_s, int *interp1_k) {
  if (0 == CheckEvent(zero_crossings->number_of_negatives - 2) *
      CheckEvent(zero_crossings->number_of_positives - 2) *
      CheckEvent(zero_crossings->number_of_peaks - 2) *
      CheckEvent(zero_crossings->number_of_dips - 2)) {
    for (int i = 0; i < f0_length; ++i) f0_candidate[i] = 0.0;
    return;
  }

  interp1a(zero_crossings->negative_interval_locations,
      zero_crossings->negative_intervals, zero_crossings->number_of_negatives,
      temporal_positions, f0_length, interpolated_f0_set[0],
      interp1_s, interp1_k);
  interp1a(zero_crossings->positive_interval_locations,
      zero_crossings->positive_intervals, zero_crossings->number_of_positives,
      temporal_positions, f0_length, interpolated_f0_set[1],
      interp1_s, interp1_k);
  interp1a(zero_crossings->peak_interval_locations,
      zero_crossings->peak_intervals, zero_crossings->number_of_peaks,
      temporal_positions, f0_length, interpolated_f0_set[2],
      interp1_s, interp1_k);
  interp1a(zero_crossings->dip_interval_locations,
      zero_crossings->dip_intervals, zero_crossings->number_of_dips,
      temporal_positions, f0_length, interpolated_f0_set[3],
      interp1_s, interp1_k);

  double upper = boundary_f0 * 1.1;
  double lower = boundary_f0 * 0.9;
  for (int i = 0; i < f0_length; ++i) {
    f0_candidate[i] = (interpolated_f0_set[0][i] +
                       interpolated_f0_set[1][i] + interpolated_f0_set[2][i] +
                       interpolated_f0_set[3][i]) / 4.0;

    if (f0_candidate[i] > upper || f0_candidate[i] < lower ||
        f0_candidate[i] > option->f0_ceil || f0_candidate[i] < option->f0_floor)
      f0_candidate[i] = 0.0;
  }
}

//-----------------------------------------------------------------------------
// DestroyZeroCrossings() frees the memory of array in the struct
//-----------------------------------------------------------------------------
static void DestroyZeroCrossings(ZeroCrossings *zero_crossings) {
  delete[] zero_crossings->fine_edges;
  delete[] zero_crossings->edges;
  delete[] zero_crossings->negative_going_points;
  delete[] zero_crossings->negative_interval_locations;
  delete[] zero_crossings->positive_interval_locations;
  delete[] zero_crossings->peak_interval_locations;
  delete[] zero_crossings->dip_interval_locations;
  delete[] zero_crossings->negative_intervals;
  delete[] zero_crossings->positive_intervals;
  delete[] zero_crossings->peak_intervals;
  delete[] zero_crossings->dip_intervals;
}

//-----------------------------------------------------------------------------
// GetRawF0Candidates() calculates f0 candidates in all channels.
//-----------------------------------------------------------------------------
static void GetRawF0Candidates(const double *boundary_f0_list,
    int number_of_bands, double actual_fs, int y_length,
    const double *temporal_positions, int f0_length,
    const fft_complex *y_spectrum, int fft_size, const HarvestOption *option,
    double *const *raw_f0_candidates) {

  GetRawF0Temp *tmps = new GetRawF0Temp[option->num_thread];
  for (int thread_id = 0; thread_id < option->num_thread; thread_id++) {
    GetRawF0Temp &tmp = tmps[thread_id];

    InitializeZeroCrossings(&tmp.zero_crossings, y_length);
    tmp.filtered_signal = new double[fft_size];
    tmp.band_pass_filter = new double[fft_size];
    tmp.band_pass_filter_spectrum = new fft_complex[fft_size];

    tmp.interp1_s = new double[f0_length];
    tmp.interp1_k = new int[f0_length];
    tmp.interpolated_f0_set = Create2DArray<world_t>(4, f0_length);
  }

#ifdef _OPENMP
#pragma omp parallel for num_threads(option->num_thread)
#endif
  for (int i = 0; i < number_of_bands; ++i) {
#ifdef _OPENMP
    int thread_id = omp_get_thread_num();
#else
    const int thread_id = 0;
#endif
    GetRawF0Temp &tmp = tmps[thread_id];

    double boundary_f0 = boundary_f0_list[i];

    GetFilteredSignal(boundary_f0, fft_size, actual_fs, y_spectrum,
      y_length, tmp.filtered_signal,
      tmp.band_pass_filter, tmp.band_pass_filter_spectrum);

    GetFourZeroCrossingIntervals(tmp, y_length, actual_fs,
      &tmp.zero_crossings);

    GetF0CandidateContour(&tmp.zero_crossings, boundary_f0, option,
      temporal_positions, f0_length, raw_f0_candidates[i],
      tmp.interpolated_f0_set, tmp.interp1_s, tmp.interp1_k);
  }

  for (int thread_id = option->num_thread-1; thread_id >= 0; thread_id--) {
    GetRawF0Temp &tmp = tmps[thread_id];

    Delete2DArray<world_t>(tmp.interpolated_f0_set);
    delete[] tmp.interp1_k;
    delete[] tmp.interp1_s;
    delete[] tmp.band_pass_filter_spectrum;
    delete[] tmp.band_pass_filter;
    delete[] tmp.filtered_signal;
    DestroyZeroCrossings(&tmp.zero_crossings);
  }
  delete[] tmps;
}

//-----------------------------------------------------------------------------
// DetectF0CandidatesSub1() calculates VUV areas.
//-----------------------------------------------------------------------------
static int DetectOfficialF0CandidatesSub1(const int *vuv,
    int number_of_channels, int *st, int *ed) {
  int number_of_voiced_sections = 0;
  for (int i = 1; i < number_of_channels; ++i) {
    int tmp = vuv[i] - vuv[i - 1];
    if (tmp == 1) st[number_of_voiced_sections] = i;
    if (tmp == -1) ed[number_of_voiced_sections++] = i;
  }

  return number_of_voiced_sections;
}

//-----------------------------------------------------------------------------
// DetectOfficialF0CandidatesSub2() calculates F0 candidates in a frame
//-----------------------------------------------------------------------------
static int DetectOfficialF0CandidatesSub2(const int *vuv,
    const double *const *raw_f0_candidates, int index,
    int number_of_voiced_sections, const int *st, const int *ed,
    const HarvestOption *option, int max_candidates, double *f0_list) {
  int number_of_candidates = 0;
  for (int i = 0; i < number_of_voiced_sections; ++i) {
    if (ed[i] - st[i] < option->f0_sub2) continue;

    double tmp_f0 = 0.0;
    for (int j = st[i]; j < ed[i]; ++j)
      tmp_f0 += raw_f0_candidates[j][index];
    tmp_f0 /= (ed[i] - st[i]);
    f0_list[number_of_candidates++] = tmp_f0;
  }

  for (int i = number_of_candidates; i < max_candidates; ++i) f0_list[i] = 0.0;
  return number_of_candidates;
}

//-----------------------------------------------------------------------------
// DetectOfficialF0Candidates() detectes F0 candidates from multi-channel
// candidates.
//-----------------------------------------------------------------------------
static int DetectOfficialF0Candidates(const double *const * raw_f0_candidates,
    int number_of_channels, int f0_length, const HarvestOption *option,
    int max_candidates, double *const *f0_candidates) {
  int number_of_candidates = 0;

  int *vuv = new int[number_of_channels];
  int *st = new int[number_of_channels];
  int *ed = new int[number_of_channels];
  for (int i = 0; i < f0_length; ++i) {
    for (int j = 0; j < number_of_channels; ++j)
      vuv[j] = raw_f0_candidates[j][i] > 0 ? 1 : 0;
    vuv[0] = vuv[number_of_channels - 1] = 0;
    int number_of_voiced_sections = DetectOfficialF0CandidatesSub1(vuv,
      number_of_channels, st, ed);
    number_of_candidates = MyMaxInt(number_of_candidates,
      DetectOfficialF0CandidatesSub2(vuv, raw_f0_candidates, i,
        number_of_voiced_sections, st, ed, option, max_candidates, f0_candidates[i]));
  }

  delete[] ed;
  delete[] st;
  delete[] vuv;
  return number_of_candidates;
}

//-----------------------------------------------------------------------------
// OverlapF0Candidates() spreads the candidates to anteroposterior frames.
//-----------------------------------------------------------------------------
static inline void OverlapF0Candidates(int f0_length, int number_of_candidates,
    double *const *f0_candidates) {
  int n = 3;
  for (int i = 1; i <= n; ++i)
    for (int j = 0; j < number_of_candidates; ++j) {
      for (int k = i; k < f0_length; ++k)
        f0_candidates[k][j + (number_of_candidates * i)] =
          f0_candidates[k - i][j];
      for (int k = 0; k < f0_length - i; ++k)
        f0_candidates[k][j + (number_of_candidates * (i + n))] =
          f0_candidates[k + i][j];
    }
}

//-----------------------------------------------------------------------------
// GetBaseIndex() calculates the temporal positions for windowing.
//-----------------------------------------------------------------------------
static inline void GetBaseIndex(double current_position, const double *base_time,
    int base_time_length, double fs, int *base_index) {
  // First-aid treatment
  int basic_index =
    lround((current_position + base_time[0]) * fs + 0.001);

  for (int i = 0; i < base_time_length; ++i) base_index[i] = basic_index + i;
}

//-----------------------------------------------------------------------------
// GetMainWindow() generates the window function.
//-----------------------------------------------------------------------------
static inline void GetMainWindow(double current_position, const int *base_index,
    int base_time_length, double fs, double window_length_in_time,
    double *main_window) {
  for (int i = 0; i < base_time_length; ++i) {
    double tmp = (base_index[i] - 1.0) / fs - current_position;
    tmp = 2.0 * world::kPi * tmp / window_length_in_time;
    main_window[i] = 0.42 + 0.5 * cos(tmp) + 0.08 * cos(2.0 * tmp);
  }
}

//-----------------------------------------------------------------------------
// GetDiffWindow() generates the differentiated window.
// Diff means differential.
//-----------------------------------------------------------------------------
static inline void GetDiffWindow(const double *main_window, int base_time_length,
    double *diff_window) {
  diff_window[0] = -main_window[1] / 2.0;
  for (int i = 1; i < base_time_length - 1; ++i)
    diff_window[i] = -(main_window[i + 1] - main_window[i - 1]) / 2.0;
  diff_window[base_time_length - 1] = main_window[base_time_length - 2] / 2.0;
}

//-----------------------------------------------------------------------------
// GetSpectra() calculates two spectra of the waveform windowed by windows
// (main window and diff window).
//-----------------------------------------------------------------------------
static inline void GetSpectra(const double *x, int x_length, int fft_size,
    const int *base_index, const double *main_window,
    const double *diff_window, int base_time_length,
    const ForwardRealFFT *forward_real_fft, fft_complex *main_spectrum,
    fft_complex *diff_spectrum, int *safe_index) {

  for (int i = 0; i < base_time_length; ++i)
    safe_index[i] = MyMaxInt(0, MyMinInt(x_length - 1, base_index[i] - 1));
  for (int i = 0; i < base_time_length; ++i)
    forward_real_fft->waveform[i] = x[safe_index[i]] * main_window[i];
  for (int i = base_time_length; i < fft_size; ++i)
    forward_real_fft->waveform[i] = 0.0;

  fft_execute(forward_real_fft->forward_fft);
  for (int i = 0; i <= fft_size / 2; ++i) {
    main_spectrum[i][0] = forward_real_fft->spectrum[i][0];
    main_spectrum[i][1] = forward_real_fft->spectrum[i][1];
  }

  for (int i = 0; i < base_time_length; ++i)
    forward_real_fft->waveform[i] = x[safe_index[i]] * diff_window[i];
  for (int i = base_time_length; i < fft_size; ++i)
    forward_real_fft->waveform[i] = 0.0;
  fft_execute(forward_real_fft->forward_fft);
  for (int i = 0; i <= fft_size / 2; ++i) {
    diff_spectrum[i][0] = forward_real_fft->spectrum[i][0];
    diff_spectrum[i][1] = forward_real_fft->spectrum[i][1];
  }
}

static inline void FixF0(const double *power_spectrum, const double *numerator_i,
    int fft_size, double fs, double current_f0, int number_of_harmonics,
    double &refined_f0, double &score) {
  double amplitude_list[NUM_HARMONICS];
  double instantaneous_frequency_list[NUM_HARMONICS];

  for (int i = 0; i < number_of_harmonics; ++i) {
    const int index = lround(current_f0 * fft_size / fs * (i + 1));
    instantaneous_frequency_list[i] = power_spectrum[index] == 0.0 ? 0.0 :
      static_cast<double>(index) * fs / fft_size +
      numerator_i[index] / power_spectrum[index] * fs / 2.0 / world::kPi;
    amplitude_list[i] = sqrt(power_spectrum[index]);
  }
  double denominator = 0.0;
  double numerator = 0.0;
  score = 0.0;
  for (int i = 0; i < number_of_harmonics; ++i) {
    numerator += amplitude_list[i] * instantaneous_frequency_list[i];
    denominator += amplitude_list[i] * (i + 1.0);
    score += fabs((instantaneous_frequency_list[i] / (i + 1.0) - current_f0) /
      current_f0);
  }

  refined_f0 = numerator / (denominator + world::kMySafeGuardMinimum);
  score = 1.0 / (score / number_of_harmonics + world::kMySafeGuardMinimum);
}

//-----------------------------------------------------------------------------
// RefineF0() modifies the F0 by instantaneous frequency.
//-----------------------------------------------------------------------------
static void RefineF0Candidates(const double *x, int x_length, double fs,
    const double *temporal_positions, int f0_length, int max_candidates,
    const HarvestOption *option,
    double *const *refined_f0_candidates, double *const *f0_scores) {

  const int max_half_window_length = static_cast<int>(1.5 * fs / option->f0_floor + 1.0);
  const int max_fft_index = 2 + static_cast<int>(log(max_half_window_length * 2 + 1.0) / world::kLog2);
  const int max_fft_size = 1 << max_fft_index;
  const int max_base_time_length = max_half_window_length * 2 + 1;
  const int min_half_window_length = static_cast<int>(1.5 * fs / option->f0_ceil + 1.0);
  const int min_fft_index = 2 + static_cast<int>(log(min_half_window_length * 2 + 1.0) / world::kLog2);
  const int num_fft = max_fft_index - min_fft_index + 1;

  RefineF0Temp *tmps = new RefineF0Temp[option->num_thread];
  for (int thread_id = 0; thread_id < option->num_thread; thread_id++) {
    RefineF0Temp &tmp = tmps[thread_id];
    tmp.main_spectrum = new fft_complex[max_fft_size];
    tmp.diff_spectrum = new fft_complex[max_fft_size];
    tmp.base_index = new int[max_base_time_length];
    tmp.main_window = new double[max_base_time_length];
    tmp.diff_window = new double[max_base_time_length];
    tmp.power_spectrum = new double[(max_fft_size / 2 + 1)];
    tmp.numerator_i = new double[(max_fft_size / 2 + 1)];
    tmp.base_time = new double[max_base_time_length];
    tmp.safe_index = new int[max_base_time_length];

    tmp.forward_real_fft_list = new ForwardRealFFT[num_fft];
    for (int i = 0; i < num_fft; i++) {
      const int fft_size = 1 << (min_fft_index+i);
      InitializeForwardRealFFT(fft_size, &tmp.forward_real_fft_list[i]);
    }
  }

#ifdef _OPENMP
#pragma omp parallel for num_threads(option->num_thread)
#endif
  for (int i = 0; i < f0_length; i++) {
#ifdef _OPENMP
    int thread_id = omp_get_thread_num();
#else
    const int thread_id = 0;
#endif
    RefineF0Temp &tmp = tmps[thread_id];
    const double current_position = temporal_positions[i];

    for (int j = 0; j < max_candidates; ++j) {
      double &refined_f0 = refined_f0_candidates[i][j];
      double &refined_score = f0_scores[i][j];
      const double current_f0 = refined_f0;

      if (current_f0 <= 0.0) {
        refined_f0 = 0.0;
        refined_score = 0.0;
        continue;
      }

      const int half_window_length = static_cast<int>(1.5 * fs / current_f0 + 1.0);
      const double window_length_in_time = (2.0 * half_window_length + 1.0) / fs;
      for (int i = 0; i < half_window_length * 2 + 1; i++)
        tmp.base_time[i] = (-half_window_length + i) / fs;
      const int fft_index = 2 + static_cast<int>(log(half_window_length * 2.0 + 1.0) / world::kLog2);
      const int fft_size = 1 << fft_index;
      const int base_time_length = half_window_length * 2 + 1;

      // calculates the instantaneous frequency
      ForwardRealFFT &forward_real_fft = tmp.forward_real_fft_list[fft_index-min_fft_index];

      GetBaseIndex(current_position, tmp.base_time, base_time_length, fs, tmp.base_index);
      GetMainWindow(current_position, tmp.base_index, base_time_length, fs,
                    window_length_in_time, tmp.main_window);
      GetDiffWindow(tmp.main_window, base_time_length, tmp.diff_window);

      GetSpectra(x, x_length, fft_size, tmp.base_index, tmp.main_window, tmp.diff_window,
                 base_time_length, &forward_real_fft, tmp.main_spectrum, tmp.diff_spectrum, tmp.safe_index);

      for (int j = 0; j <= fft_size / 2; ++j) {
        tmp.numerator_i[j] = tmp.main_spectrum[j][0] * tmp.diff_spectrum[j][1] -
                tmp.main_spectrum[j][1] * tmp.diff_spectrum[j][0];
        tmp.power_spectrum[j] = tmp.main_spectrum[j][0] * tmp.main_spectrum[j][0] +
                tmp.main_spectrum[j][1] * tmp.main_spectrum[j][1];
      }

      const int number_of_harmonics =
              MyMinInt(static_cast<int>(fs / 2.0 / current_f0), NUM_HARMONICS);
      FixF0(tmp.power_spectrum, tmp.numerator_i, fft_size, fs, current_f0,
            number_of_harmonics, refined_f0, refined_score);

      if (refined_f0 < option->f0_floor || refined_f0 > option->f0_ceil ||
          refined_score < 2.5) {
          refined_f0 = 0.0;
          refined_score = 0.0;
      }
    }
  }

  for (int thread_id = option->num_thread-1; thread_id >= 0; thread_id--) {
    RefineF0Temp &tmp = tmps[thread_id];

    for (int i = num_fft-1; i >= 0; i--) {
      DestroyForwardRealFFT(&tmp.forward_real_fft_list[i]);
    }
    delete[] tmp.forward_real_fft_list;

    delete[] tmp.safe_index;
    delete[] tmp.base_time;
    delete[] tmp.numerator_i;
    delete[] tmp.power_spectrum;
    delete[] tmp.diff_window;
    delete[] tmp.main_window;
    delete[] tmp.base_index;
    delete[] tmp.diff_spectrum;
    delete[] tmp.main_spectrum;
  }
  delete[] tmps;
}

//-----------------------------------------------------------------------------
// SelectBestF0() obtains the nearlest F0 in reference_f0.
//-----------------------------------------------------------------------------
static double SelectBestF0(double reference_f0, const double *f0_candidates,
    int number_of_candidates, double allowed_range, double &best_error) {
  double best_f0 = 0.0;
  best_error = allowed_range;

  double tmp;
  for (int i = 0; i < number_of_candidates; ++i) {
    tmp = fabs(reference_f0 - f0_candidates[i]) / reference_f0;
    if (tmp > best_error) continue;
    best_f0 = f0_candidates[i];
    best_error = tmp;
  }

  return best_f0;
}

static void RemoveUnreliableCandidatesSub(int i, int j,
    const double *const *tmp_f0_candidates, int number_of_candidates,
    double *const *f0_candidates, double *const *f0_scores) {
  double reference_f0 = f0_candidates[i][j];
  double error1, error2, min_error;
  double threshold = 0.05;
  if (reference_f0 == 0) return;
  SelectBestF0(reference_f0, tmp_f0_candidates[i + 1],
      number_of_candidates, 1.0, error1);
  SelectBestF0(reference_f0, tmp_f0_candidates[i - 1],
      number_of_candidates, 1.0, error2);
  min_error = fmin(error1, error2);
  if (min_error <= threshold) return;
  f0_candidates[i][j] = 0;
  f0_scores[i][j] = 0;
}

//-----------------------------------------------------------------------------
// RemoveUnreliableCandidates().
//-----------------------------------------------------------------------------
static void RemoveUnreliableCandidates(int f0_length, int number_of_candidates,
    double *const *f0_candidates, double *const *f0_scores) {

  double **tmp_f0_candidates = Create2DArray<double>(f0_length, number_of_candidates);
  for (int i = 0; i < f0_length; ++i)
    for (int j = 0; j < number_of_candidates; ++j)
      tmp_f0_candidates[i][j] = f0_candidates[i][j];

  for (int i = 1; i < f0_length - 1; ++i)
    for (int j = 0; j < number_of_candidates; ++j)
      RemoveUnreliableCandidatesSub(i, j, tmp_f0_candidates,
          number_of_candidates, f0_candidates, f0_scores);

  Delete2DArray(tmp_f0_candidates);
}

//-----------------------------------------------------------------------------
// SearchF0Base() gets the F0 with the highest score.
//-----------------------------------------------------------------------------
static void SearchF0Base(const double *const *f0_candidates,
    const double *const *f0_scores, int f0_length, int number_of_candidates,
    double *base_f0_contour) {
  double tmp_best_score;
  for (int i = 0; i < f0_length; ++i) {
    base_f0_contour[i] = tmp_best_score = 0.0;
    for (int j = 0; j < number_of_candidates; ++j)
      if (f0_scores[i][j] > tmp_best_score) {
        base_f0_contour[i] = f0_candidates[i][j];
        tmp_best_score = f0_scores[i][j];
      }
  }
}

//-----------------------------------------------------------------------------
// Step 1: Rapid change of F0 contour is replaced by 0.
//-----------------------------------------------------------------------------
static void FixStep1(const double *f0_base, int f0_length,
    double allowed_range, double *f0_step1) {
  for (int i = 0; i < f0_length; ++i) f0_step1[i] = 0.0;
  double reference_f0;
  for (int i = 2; i < f0_length; ++i) {
    if (f0_base[i] == 0.0) continue;
    reference_f0 = f0_base[i - 1] * 2 - f0_base[i - 2];
    f0_step1[i] =
      fabs((f0_base[i] - reference_f0) / reference_f0) > allowed_range &&
      fabs((f0_base[i] - f0_base[i - 1])) / f0_base[i - 1] > allowed_range ?
      0.0 : f0_base[i];
  }
}

//-----------------------------------------------------------------------------
// GetBoundaryList() detects boundaries between voiced and unvoiced sections.
//-----------------------------------------------------------------------------
static int GetBoundaryList(const double *f0, int f0_length,
    int *boundary_list) {
  int number_of_boundaries = 0;
  int *vuv = new int[f0_length];
  for (int i = 0; i < f0_length; ++i)
    vuv[i] = f0[i] > 0 ? 1 : 0;
  vuv[0] = vuv[f0_length - 1] = 0;

  for (int i = 1; i < f0_length; ++i)
    if (vuv[i] - vuv[i - 1] != 0) {
      boundary_list[number_of_boundaries] = i - number_of_boundaries % 2;
      number_of_boundaries++;
    }

  delete[] vuv;
  return number_of_boundaries;
}

//-----------------------------------------------------------------------------
// Step 2: Voiced sections with a short period are removed.
//-----------------------------------------------------------------------------
static void FixStep2(const double *f0_step1, int f0_length,
    int voice_range_minimum, double *f0_step2) {
  for (int i = 0; i < f0_length; ++i) f0_step2[i] = f0_step1[i];
  int *boundary_list = new int[f0_length];
  int number_of_boundaries =
    GetBoundaryList(f0_step1, f0_length, boundary_list);

  for (int i = 0; i < number_of_boundaries / 2; ++i) {
    if (boundary_list[i * 2 + 1] - boundary_list[i * 2] >= voice_range_minimum)
      continue;
    for (int j = boundary_list[i * 2]; j <= boundary_list[(i * 2) + 1]; ++j)
      f0_step2[j] = 0.0;
  }
  delete[] boundary_list;
}

//-----------------------------------------------------------------------------
// GetMultiChannelF0() separates each voiced section into independent channel.
//-----------------------------------------------------------------------------
static void GetMultiChannelF0(const double *f0, int f0_length,
    const int *boundary_list, int number_of_boundaries,
    double *const *multi_channel_f0) {
  for (int i = 0; i < number_of_boundaries / 2; ++i) {
    for (int j = 0; j < boundary_list[i * 2]; ++j)
      multi_channel_f0[i][j] = 0.0;
    for (int j = boundary_list[i * 2]; j <= boundary_list[i * 2 + 1]; ++j)
      multi_channel_f0[i][j] = f0[j];
    for (int j = boundary_list[i * 2 + 1] + 1; j < f0_length; ++j)
      multi_channel_f0[i][j] = 0.0;
  }
}

//-----------------------------------------------------------------------------
// abs() often causes bugs, an original function is used.
//-----------------------------------------------------------------------------
static inline int MyAbsInt(int x) {
  return x > 0 ? x : -x;
}

//-----------------------------------------------------------------------------
// ExtendF0() : The Hand erasing the Space.
// The subfunction of Extend().
//-----------------------------------------------------------------------------
static int ExtendF0(const double *f0, int f0_length, int origin,
    int last_point, int shift, const double *const *f0_candidates,
    int number_of_candidates, double allowed_range, double *extended_f0) {
  int threshold = 4;
  double tmp_f0 = extended_f0[origin];
  int shifted_origin = origin;

  int distance = MyAbsInt(last_point - origin);
  int *index_list = new int[distance + 1];
  for (int i = 0; i <= distance; ++i) index_list[i] = origin + shift * i;

  int count = 0;
  for (int i = 0; i <= distance; ++i) {
    double dammy;
    extended_f0[index_list[i] + shift] =
      SelectBestF0(tmp_f0, f0_candidates[index_list[i] + shift],
      number_of_candidates, allowed_range, dammy);
    if (extended_f0[index_list[i] + shift] == 0.0) {
      count++;
    } else {
      tmp_f0 = extended_f0[index_list[i] + shift];
      count = 0;
      shifted_origin = index_list[i] + shift;
    }
    if (count == threshold) break;
  }

  delete[] index_list;
  return shifted_origin;
}

//-----------------------------------------------------------------------------
// Swap the f0 contour and boundary.
// It is used in Extend() and MergeF0();
//-----------------------------------------------------------------------------
static inline void Swap(int index1, int index2, int f0_length, double *const *f0, int *boundary) {
  double* f0_i1 = f0[index1];
  double* f0_i2 = f0[index2];
  for(int i=0; i < f0_length; i++) {
    std::swap(f0_i1[i], f0_i2[i]);
  }
  std::swap(boundary[index1 * 2], boundary[index2 * 2]);
  std::swap(boundary[index1 * 2 + 1], boundary[index2 * 2 + 1]);
}

//-----------------------------------------------------------------------------
// Extend() : The Hand erasing the Space.
//-----------------------------------------------------------------------------
static int Extend(const double *const *multi_channel_f0,
    int number_of_sections, int f0_length, const int *boundary_list,
    const double *const *f0_candidates, int number_of_candidates,
    double allowed_range, double *const *extended_f0, int *shifted_boundary_list) {
  int threshold = 100;
  for (int i = 0; i < number_of_sections; ++i) {
    shifted_boundary_list[i * 2 + 1] = ExtendF0(multi_channel_f0[i],
      f0_length, boundary_list[i * 2 + 1],
      MyMinInt(f0_length - 2, boundary_list[i * 2 + 1] + threshold), 1,
      f0_candidates, number_of_candidates, allowed_range, extended_f0[i]);
    shifted_boundary_list[i * 2] = ExtendF0(multi_channel_f0[i], f0_length,
      boundary_list[i * 2], MyMaxInt(1, boundary_list[i * 2] - threshold), -1,
      f0_candidates, number_of_candidates, allowed_range, extended_f0[i]);
  }

  double threshold2 = 2200.0;
  int count = 0;
  double mean_f0 = 0.0;
  int st, ed;
  for (int i = 0; i < number_of_sections; ++i) {
    st = boundary_list[i * 2];
    ed = boundary_list[i * 2 + 1];
    for (int j = st; j < ed; ++j) mean_f0 += multi_channel_f0[i][j];
    mean_f0 /= ed - st;
    if (threshold2 / mean_f0 < ed - st)
      Swap(count++, i, f0_length, extended_f0, shifted_boundary_list);
  }
  return count;
}

//-----------------------------------------------------------------------------
// Indices are sorted.
//-----------------------------------------------------------------------------
static void MakeSortedOrder(const int *boundary_list, int number_of_sections,
    int *order) {
  for (int i = 0; i < number_of_sections; ++i) order[i] = i;
  int tmp;
  for (int i = 1; i < number_of_sections; ++i)
    for (int j = i - 1; j >= 0; --j)
      if (boundary_list[order[j] * 2] > boundary_list[order[i] * 2]) {
        tmp = order[i];
        order[i] = order[j];
        order[j] = tmp;
      } else {
        break;
      }
}

//-----------------------------------------------------------------------------
// Serach the highest score with the candidate F0.
//-----------------------------------------------------------------------------
static inline double SearchScore(double f0, const double *f0_candidates,
    const double *f0_scores, int number_of_candidates) {
  double score = 0.0;
  for (int i = 0; i < number_of_candidates; ++i)
    if (f0 == f0_candidates[i] && score < f0_scores[i]) score = f0_scores[i];
  return score;
}

//-----------------------------------------------------------------------------
// Subfunction of MergeF0()
//-----------------------------------------------------------------------------
static int MergeF0Sub(const double *f0_1, int f0_length, int st1, int ed1,
    const double *f0_2, int st2, int ed2, const double *const *f0_candidates,
    const double *const *f0_scores, int number_of_candidates,
    double *merged_f0) {
  if (st1 <= st2 && ed1 >= ed2) return ed1;

  double score1 = 0.0;
  double score2 = 0.0;
  for (int i = st2; i <= ed1; ++i) {
    score1 += SearchScore(f0_1[i], f0_candidates[i], f0_scores[i],
      number_of_candidates);
    score2 += SearchScore(f0_2[i], f0_candidates[i], f0_scores[i],
      number_of_candidates);
  }
  if (score1 > score2)
    for (int i = ed1; i <= ed2; ++i) merged_f0[i] = f0_2[i];
  else
    for (int i = st2; i <= ed2; ++i) merged_f0[i] = f0_2[i];

  return ed2;
}

//-----------------------------------------------------------------------------
// Overlapped F0 contours are merged by the likability score.
//-----------------------------------------------------------------------------
static void MergeF0(const double *const *multi_channel_f0, int *boundary_list,
    int number_of_channels, int f0_length, const double *const *f0_candidates,
    const double *const *f0_scores, int number_of_candidates,
    double *merged_f0) {
  int *order = new int[number_of_channels];
  MakeSortedOrder(boundary_list, number_of_channels, order);

  for (int i = 0; i < f0_length; ++i)
    merged_f0[i] = multi_channel_f0[0][i];

  for (int i = 1; i < number_of_channels; ++i)
    if (boundary_list[order[i] * 2] - boundary_list[1] > 0) {
      for (int j = boundary_list[order[i] * 2];
        j <= boundary_list[order[i] * 2 + 1]; ++j)
        merged_f0[j] = multi_channel_f0[order[i]][j];
      boundary_list[0] = boundary_list[order[i] * 2];
      boundary_list[1] = boundary_list[order[i] * 2 + 1];
    } else {
      boundary_list[1] =
        MergeF0Sub(merged_f0, f0_length, boundary_list[0], boundary_list[1],
        multi_channel_f0[order[i]], boundary_list[order[i] * 2],
        boundary_list[order[i] * 2 + 1], f0_candidates, f0_scores,
        number_of_candidates, merged_f0);
    }

  delete[] order;
}

//-----------------------------------------------------------------------------
// Step 3: Voiced sections are extended based on the continuity of F0 contour
//-----------------------------------------------------------------------------
static void FixStep3(const double *f0_step2, int f0_length,
    int number_of_candidates, const double *const *f0_candidates,
    double allowed_range, const double *const *f0_scores, double *f0_step3) {
  for (int i = 0; i < f0_length; ++i) f0_step3[i] = f0_step2[i];
  int *boundary_list = new int[f0_length];
  int number_of_boundaries =
    GetBoundaryList(f0_step2, f0_length, boundary_list);

  double **multi_channel_f0 = Create2DArray<double>(number_of_boundaries, f0_length);
  GetMultiChannelF0(f0_step2, f0_length, boundary_list, number_of_boundaries,
      multi_channel_f0);

  int number_of_channels =
    Extend(multi_channel_f0, number_of_boundaries / 2, f0_length,
    boundary_list, f0_candidates, number_of_candidates, allowed_range,
    multi_channel_f0, boundary_list);

  if (number_of_channels != 0)
    MergeF0(multi_channel_f0, boundary_list, number_of_channels, f0_length,
        f0_candidates, f0_scores, number_of_candidates, f0_step3);

  Delete2DArray(multi_channel_f0);
  delete[] boundary_list;
}

//-----------------------------------------------------------------------------
// Step 4: F0s in short unvoiced section are faked
//-----------------------------------------------------------------------------
static void FixStep4(const double *f0_step3, int f0_length, int threshold,
    double *f0_step4) {
  for (int i = 0; i < f0_length; ++i) f0_step4[i] = f0_step3[i];
  int *boundary_list = new int[f0_length];
  int number_of_boundaries =
    GetBoundaryList(f0_step3, f0_length, boundary_list);

  int distance;
  double tmp0, tmp1, coefficient;
  int count;
  for (int i = 0; i < number_of_boundaries / 2 - 1; ++i) {
    distance = boundary_list[(i + 1) * 2] - boundary_list[i * 2 + 1] - 1;
    if (distance >= threshold) continue;
    tmp0 = f0_step3[boundary_list[i * 2 + 1]] + 1;
    tmp1 = f0_step3[boundary_list[(i + 1) * 2]] - 1;
    coefficient = (tmp1 - tmp0) / (distance + 1.0);
    count = 1;
    for (int j = boundary_list[i * 2 + 1] + 1;
        j <= boundary_list[(i + 1) * 2] - 1; ++j)
      f0_step4[j] = tmp0 + coefficient * count++;
  }
  delete[] boundary_list;
}

//-----------------------------------------------------------------------------
// FixF0Contour() obtains the likely F0 contour.
//-----------------------------------------------------------------------------
static void FixF0Contour(const double *const *f0_candidates,
    const double *const *f0_scores, int f0_length, int number_of_candidates,
    double *best_f0_contour) {
  double *tmp_f0_contour1 = new double[f0_length];
  double *tmp_f0_contour2 = new double[f0_length];

  // These parameters are optimized by speech databases.
  SearchF0Base(f0_candidates, f0_scores, f0_length,
      number_of_candidates, tmp_f0_contour1);
  FixStep1(tmp_f0_contour1, f0_length, 0.008, tmp_f0_contour2);
  FixStep2(tmp_f0_contour2, f0_length, 6, tmp_f0_contour1);
  FixStep3(tmp_f0_contour1, f0_length, number_of_candidates, f0_candidates,
      0.18, f0_scores, tmp_f0_contour2);
  FixStep4(tmp_f0_contour2, f0_length, 9, best_f0_contour);

  delete[] tmp_f0_contour2;
  delete[] tmp_f0_contour1;
}

//-----------------------------------------------------------------------------
// This function uses zero-lag Butterworth filter.
//-----------------------------------------------------------------------------
static void FilteringF0(const double *a, const double *b, double *x,
    int x_length, int st, int ed, double *y) {
  double w[2] = { 0.0, 0.0 };
  double wt;
  double *tmp_x = new double[x_length];

  for (int i = 0; i < st; ++i) x[i] = x[st];
  for (int i = ed + 1; i < x_length; ++i) x[i] = x[ed];

  for (int i = 0; i < x_length; ++i) {
    wt = x[i] + a[0] * w[0] + a[1] * w[1];
    tmp_x[x_length - i - 1] = b[0] * wt + b[1] * w[0] + b[0] * w[1];
    w[1] = w[0];
    w[0] = wt;
  }

  w[0] = w[1] = 0.0;
  for (int i = 0; i < x_length; ++i) {
    wt = tmp_x[i] + a[0] * w[0] + a[1] * w[1];
    y[x_length - i - 1] = b[0] * wt + b[1] * w[0] + b[0] * w[1];
    w[1] = w[0];
    w[0] = wt;
  }

  delete[] tmp_x;
}

//-----------------------------------------------------------------------------
// SmoothF0Contour() uses the zero-lag Butterworth filter for smoothing.
//-----------------------------------------------------------------------------
static void SmoothF0Contour(const double *f0, int f0_length,
    double *smoothed_f0) {
  const double b[2] =
    { 0.0078202080334971724, 0.015640416066994345 };
  const double a[2] =
    { 1.7347257688092754, -0.76600660094326412 };
  int lag = 300;
  int new_f0_length = f0_length + lag * 2;
  double *f0_contour = new double[new_f0_length];
  for (int i = 0; i < lag; ++i) f0_contour[i] = 0.0;
  for (int i = lag; i < lag + f0_length; ++i) f0_contour[i] = f0[i - lag];
  for (int i = lag + f0_length; i < new_f0_length; ++i) f0_contour[i] = 0.0;

  int *boundary_list = new int[new_f0_length];
  int number_of_boundaries =
    GetBoundaryList(f0_contour, new_f0_length, boundary_list);
  double **multi_channel_f0 = Create2DArray<double>(number_of_boundaries / 2, new_f0_length);
  GetMultiChannelF0(f0_contour, new_f0_length, boundary_list,
      number_of_boundaries, multi_channel_f0);

  for (int i = 0; i < number_of_boundaries / 2; ++i) {
    FilteringF0(a, b, multi_channel_f0[i], new_f0_length,
      boundary_list[i * 2], boundary_list[i * 2 + 1], f0_contour);
    for (int j = boundary_list[i * 2]; j <= boundary_list[i * 2 + 1]; ++j)
      smoothed_f0[j - lag] = f0_contour[j];
  }

  Delete2DArray(multi_channel_f0);
  delete[] boundary_list;
  delete[] f0_contour;
}

//-----------------------------------------------------------------------------
// HarvestGeneralBodySub() is the subfunction of HarvestGeneralBody()
//-----------------------------------------------------------------------------
static int HarvestGeneralBodySub(const double *boundary_f0_list,
    int number_of_channels, int f0_length, double actual_fs, int y_length,
    const double *temporal_positions, const fft_complex *y_spectrum,
    int fft_size, const HarvestOption *option, int max_candidates,
    double *const *f0_candidates) {

  double **raw_f0_candidates = Create2DArray<double>(number_of_channels, f0_length);

  GetRawF0Candidates(boundary_f0_list, number_of_channels,
      actual_fs, y_length, temporal_positions, f0_length, y_spectrum,
      fft_size, option, raw_f0_candidates);

  int number_of_candidates = DetectOfficialF0Candidates(raw_f0_candidates,
    number_of_channels, f0_length, option, max_candidates, f0_candidates);

  OverlapF0Candidates(f0_length, number_of_candidates, f0_candidates);

  Delete2DArray(raw_f0_candidates);
  return number_of_candidates;
}

//-----------------------------------------------------------------------------
// HarvestGeneralBody() estimates the F0 contour based on Harvest.
//-----------------------------------------------------------------------------
static void HarvestGeneralBody(const double *x, int x_length, int fs,
    int frame_period, const HarvestOption *option, int speed, double *temporal_positions,
    double *f0) {
  double adjusted_f0_floor = option->f0_floor * 0.9;
  double adjusted_f0_ceil = option->f0_ceil * 1.1;
  int number_of_channels =
    1 + static_cast<int>(log(adjusted_f0_ceil / adjusted_f0_floor) /
    world::kLog2 * option->channels_in_octave);
  double *boundary_f0_list = new double[number_of_channels];
  for (int i = 0; i < number_of_channels; ++i)
    boundary_f0_list[i] =
            adjusted_f0_floor * pow(2.0, (i + 1) / option->channels_in_octave);

  // normalization
  int decimation_ratio = MyMaxInt(MyMinInt(speed, 12), 1);
  int y_length =
    static_cast<int>(ceil(static_cast<double>(x_length) / decimation_ratio));
  double actual_fs = static_cast<double>(fs) / decimation_ratio;
  int fft_size = GetSuitableFFTSize(y_length + 5 +
    2 * static_cast<int>(2.0 * actual_fs / boundary_f0_list[0]));

  // Calculation of the spectrum used for the f0 estimation
  double *y = new double[fft_size];
  fft_complex *y_spectrum = new fft_complex[fft_size];
  GetWaveformAndSpectrum(x, x_length, y_length, actual_fs, fft_size,
      decimation_ratio, y, y_spectrum);

  int f0_length = GetSamplesForHarvest(fs, x_length, frame_period);
  for (int i = 0; i < f0_length; ++i) {
    temporal_positions[i] = i * frame_period / 1000.0;
    f0[i] = 0.0;
  }

  int overlap_parameter = 7;
  int max_candidates =
    lround(number_of_channels / 10.0) * overlap_parameter;
  double **f0_candidates = Create2DArray<double>(f0_length, max_candidates);
  double **f0_candidates_score = Create2DArray<double>(f0_length, max_candidates);

  int number_of_candidates = HarvestGeneralBodySub(boundary_f0_list,
    number_of_channels, f0_length, actual_fs, y_length, temporal_positions,
    y_spectrum, fft_size, option, max_candidates, f0_candidates) *
    overlap_parameter;

  RefineF0Candidates(y, y_length, actual_fs, temporal_positions, f0_length,
      number_of_candidates, option, f0_candidates,
      f0_candidates_score);
  RemoveUnreliableCandidates(f0_length, number_of_candidates,
      f0_candidates, f0_candidates_score);

  double *best_f0_contour = new double[f0_length];
  FixF0Contour(f0_candidates, f0_candidates_score, f0_length,
      number_of_candidates, best_f0_contour);
  SmoothF0Contour(best_f0_contour, f0_length, f0);

  delete[] best_f0_contour;
  Delete2DArray(f0_candidates_score);
  Delete2DArray(f0_candidates);
  delete[] y_spectrum;
  delete[] y;
  delete[] boundary_f0_list;
}

}  // namespace

int GetSamplesForHarvest(int fs, int x_length, double frame_period) {
  return static_cast<int>(1000.0 * x_length / fs / frame_period) + 1;
}

void Harvest(const double *x, int x_length, int fs,
    const HarvestOption *option, double *temporal_positions, double *f0) {
  int dimension_ratio = lround(fs / option->target_fs);

  if (option->frame_period == 1.0) {
    HarvestGeneralBody(x, x_length, fs, 1, option, dimension_ratio,
        temporal_positions, f0);
    return;
  }

  int basic_frame_period = 1;
  int basic_f0_length =
    GetSamplesForHarvest(fs, x_length, basic_frame_period);
  double *basic_f0 = new double[basic_f0_length];
  double *basic_temporal_positions = new double[basic_f0_length];
  HarvestGeneralBody(x, x_length, fs, basic_frame_period, option, dimension_ratio,
      basic_temporal_positions, basic_f0);

  int f0_length = GetSamplesForHarvest(fs, x_length, option->frame_period);
  for (int i = 0; i < f0_length; ++i) {
    temporal_positions[i] = i * option->frame_period / 1000.0;
    f0[i] = basic_f0[MyMinInt(basic_f0_length - 1,
      lround(temporal_positions[i] * 1000.0))];
  }

  delete[] basic_temporal_positions;
  delete[] basic_f0;
}

void InitializeHarvestOption(HarvestOption *option) {
#ifdef _OPENMP
  option->num_thread = omp_get_num_procs();
#else
  option->num_thread = 1;
#endif

  // You can change default parameters.
  option->f0_ceil = world::kCeilF0;
  option->f0_floor = world::kFloorF0;
  option->frame_period = 5;
  option->target_fs = 8000;
  option->channels_in_octave = 40;
  option->f0_sub2 = 10;
}
