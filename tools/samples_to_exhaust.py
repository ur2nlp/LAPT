import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_sizes', type=int, nargs='+')
    parser.add_argument('--alpha', type=float)
    args = parser.parse_args()

    exponiated_sizes = [size**args.alpha for size in args.input_sizes]
    exponiated_sum = sum(exponiated_sizes)
    sampling_probs = [size / exponiated_sum for size in exponiated_sizes]
    samples_to_exhaust = [int(size / prob) for size, prob in zip(args.input_sizes, sampling_probs)]

    for i in range(len(args.input_sizes)):
        print(f"Input size: {args.input_sizes[i]}")
        print(f"Sampling probability: {round(sampling_probs[i], 4)}")
        print(f"Total samples to exhaustion: {samples_to_exhaust[i]}\n")

if __name__ == '__main__':
    main()
