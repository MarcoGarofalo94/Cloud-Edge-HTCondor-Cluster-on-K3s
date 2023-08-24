## 1. Install Prometheus Stack

```
https://github.com/prometheus-community/helm-charts/tree/kube-prometheus-stack-16.0.1/charts/kube-prometheus-stack#configuration
```

## 2. Run a deployment that expose a metric
In this setup we created a go server that exposes metrics on the endpoint `/metrics/{architecture}` supporting the `amd64` and `arm64` parameters as architectures.
The server can be found at: https://github.com/MarcoGarofalo94/metric-service

The deployment in this setup points to the `htcondor-schedd` that can be found in this repo at: https://github.com/MarcoGarofalo94/simple-htcondor-cluster-on-kubernetes/blob/master/htcondor-schedd-deployment.yaml.

## 3. Run a service associated to previous deployment in order to expose the metric
To expose the custom metrics we need to attach a service to the deployment, again the service can be found at:
https://github.com/MarcoGarofalo94/simple-htcondor-cluster-on-kubernetes/blob/master/htcondor-schedd-service.yaml

## 4. Run the ServiceMonitor to monitor the deployment
Since we want to autoscale two different type of deployments, in the `htcondor-schedd-service-monitor.yaml` we can find two ServiceMonitors: the first covers the amd64 deployment while the second refers to the arm64 deployment.

```
kubectl apply -f https://raw.githubusercontent.com/MarcoGarofalo94/simple-htcondor-cluster-on-kubernetes/master/autoscaling/htcondor-schedd-service-monitor.yaml
```
If everything it's working, we should se the `amd64_cluster_utilization` and `arm64_cluster_utilization` on our Prometheus dashboard (web).

## 5. Install Prometheus Adapter
Now we need the adapter in order to make Kubernetes and Prometheus communicate seamlessly on our custom metrics.
```
https://github.com/kubernetes-sigs/prometheus-adapter/blob/master/deploy/README.md
```

We can modify the ConfigMap associated to the Prometheus adapter in order to specificy which queries to perform:
```
kubectl apply -f https://raw.githubusercontent.com/MarcoGarofalo94/simple-htcondor-cluster-on-kubernetes/master/autoscaling/prom-adapter-config.yaml
# Need to restart te prometheus adapter deployment
kubectl rollout restart deployment prometheus-adapter -n monitoring
```

Finally we need to register the custom metrics API with the API aggregator (part of the main Kubernetes API server) through an API Service resource.
```
kubectl apply -f https://raw.githubusercontent.com/MarcoGarofalo94/simple-htcondor-cluster-on-kubernetes/master/autoscaling/api-service.yaml
```

At this point, if everything is working, we can check if Kubernetes can access our custom metrics (change amd64 to arm64 to verify the second metric):
```
kubectl get --raw "/apis/custom.metrics.k8s.io/v1beta2/namespaces/default/pods/*/amd64?selector=app%3Dhtcondor-schedd"
```

## 6. Horizontal Pod Autoscaler
Now that we have a deployment exposing a couple of metrics, a prometheus instance gathering those metrics and a prometheus adapter able to communicate with Kubernetes, meaning that Kubernetes can access our custom metrics, we can setup the HorizontalPodAutoscaler resources to autoscale our deployment.

In this project we actually gather the metrics from a deployment `htcondor-schedd` but we want to scale different deployments that are `htcondor-worker` and `htcondor-worker-arm64`.

```
kubectl apply -f https://raw.githubusercontent.com/MarcoGarofalo94/simple-htcondor-cluster-on-kubernetes/master/autoscaling/amd64-hpa.yaml

kubectl apply -f https://raw.githubusercontent.com/MarcoGarofalo94/simple-htcondor-cluster-on-kubernetes/master/autoscaling/arm64-hpa.yaml
```

After few moments we should be able to inspect the logs of the HPA
```
kubectl describe hpa htcondor-schedd-amd64 
# OR htcondor-schedd-arm64 for the arm64 deployment
```
```
# We should get something like
Name:                                                 htcondor-schedd-amd64
Namespace:                                            default
Labels:                                               <none>
Annotations:                                          <none>
CreationTimestamp:                                    Thu, 24 Aug 2023 07:28:28 +0000
Reference:                                            Deployment/htcondor-worker
Metrics:                                              ( current / target )
  "amd64" on Service/htcondor-schedd (target value):  0 / 1
Min replicas:                                         1
Max replicas:                                         6
Behavior:
  Scale Up:
    Stabilization Window: 5 seconds
    Select Policy: Max
    Policies:
      - Type: Percent  Value: 100  Period: 5 seconds
  Scale Down:
    Stabilization Window: 200 seconds
    Select Policy: Max
    Policies:
      - Type: Percent  Value: 50  Period: 15 seconds
Deployment pods:       1 current / 1 desired
Conditions:
  Type            Status  Reason            Message
  ----            ------  ------            -------
  AbleToScale     True    ReadyForNewScale  recommended size matches current size
  ScalingActive   True    ValidMetricFound  the HPA was able to successfully calculate a replica count from Service metric amd64
  ScalingLimited  True    TooFewReplicas    the desired replica count is less than the minimum replica count
Events:           <none>
```
