# Young Me Up

Young Me Up is a web app that uses a deep learning model to analyze a person's age based on an image.

Through the magic of a final linear layer - the "soft-focus" layer - 10 years are subtracted from the person's age.

Experience the wizardry at https://youngmeup.now.sh/

## Deployment

```shell
now
export NAME='youngmeup'
now alias $NAME
```

## Scaling

To scale deployment:

`now scale $NAME.now.sh sfo 1`

See https://zeit.co/blog/scale

