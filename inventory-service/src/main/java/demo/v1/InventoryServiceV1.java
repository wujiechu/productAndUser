package demo.v1;


import demo.product.Product;
import demo.product.ProductRepository;
import org.neo4j.ogm.session.Session;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;


@Service
public class InventoryServiceV1 {
    private ProductRepository productRepository;
    private Session neo4jTemplate;

    @Autowired
    public InventoryServiceV1(ProductRepository productRepository, Session neo4jTemplate) {
        this.productRepository = productRepository;
        this.neo4jTemplate = neo4jTemplate;
    }

    public Product getProduct(String productId) {
        Product product;

        product = productRepository.getProductByProductId(productId);


        return product;
    }

}
